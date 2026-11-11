"""Fast mid / top selection (default on; RECSA_FAST_SELECT=0 = the TF32 selection kernels)."""
import torch
import triton
import triton.language as tl

from recsa.merge_acc import query_positions, rotate_keys


@triton.jit
def _rank_select_early(us, TOP_K: tl.constexpr):
    """Bit-bisection threshold of the TOP_K-th largest of the pairwise-distinct int32 keys us (any 2-D shape), stopping
    at the first candidate whose count is exactly TOP_K."""
    SIGN: tl.constexpr = -2147483648
    tau = tl.sum(tl.zeros([1], dtype=tl.int32), axis=0)
    done = tl.sum(tl.zeros([1], dtype=tl.int32), axis=0)
    for step in tl.static_range(32):
        if done == 0:
            if step == 0:
                cand = tau | SIGN
            else:
                cand = tau | (1 << (31 - step))
            cnt = tl.sum(tl.sum((us >= (cand ^ SIGN)).to(tl.int32), axis=1), axis=0)
            tau = tl.where(cnt >= TOP_K, cand, tau)
            done = (cnt == TOP_K).to(tl.int32)
    return tl.zeros([1], dtype=tl.int32) + (tau ^ SIGN)


@triton.jit
def _replay_chain(v, m_par, live2, slot2, ncand, n_batches, NB: tl.constexpr, LOG_N: tl.constexpr,
                  NEG_INF: tl.constexpr):
    """v: [NB, BC] raw importances relative to m_par (local max of the slot's parent / tile), row b = batch b of the
    TF32 kernels."""
    idx_mask = (1 << LOG_N) - 1
    o_b = tl.arange(0, NB)
    bm = tl.max(tl.where(live2, m_par, NEG_INF), axis=1)                                  # [NB] batch max
    v = tl.where(v >= 0, v * tl.exp(m_par - bm[:, None]), v)
    run = tl.max(tl.where(o_b[None, :] <= o_b[:, None], bm[None, :], NEG_INF), axis=1)   # running max after b
    prv = tl.max(tl.where(o_b[None, :] < o_b[:, None], bm[None, :], NEG_INF), axis=1)    # ... before b
    rb = tl.exp(bm - run)
    rp = tl.exp(prv - run)
    v = tl.where(v >= 0, v * rb[:, None], v)
    v = tl.where(slot2 == ncand - 1, 1e6, v)                                              # rightmost forced
    v = tl.where(live2, v, NEG_INF)
    v = (v.to(tl.int32, bitcast=True) & ~idx_mask).to(tl.float32, bitcast=True)
    for j in tl.static_range(1, NB):
        f_j = tl.sum(tl.where(o_b == j, rp, 0.0), axis=0)
        if (f_j != 1.0) & (j < n_batches):     # factor 1.0: multiply and re-truncate are no-ops
            v_j = tl.where(v >= 0, v * f_j, v)
            v_j = (v_j.to(tl.int32, bitcast=True) & ~idx_mask).to(tl.float32, bitcast=True)
            v = tl.where(o_b[:, None] < j, v_j, v)
    return v


@triton.jit
def mid_select_finalize_kernel(
    raw, pmax, prev_selected_parents, next_selected_parents,
    t_offset, t_base, n_prog_t, T,
    layer_power: tl.constexpr, H_kv: tl.constexpr, COMPRESSION_RATE: tl.constexpr, TOP_K: tl.constexpr,
    LOG_N: tl.constexpr, SCORE_VALID_THRESHOLD: tl.constexpr, NEG_INF: tl.constexpr,
):
    SIGN: tl.constexpr = -2147483648
    MAX_CAND: tl.constexpr = TOP_K * COMPRESSION_RATE
    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE
    NTILE: tl.constexpr = MAX_CAND // BC
    pid_local = tl.program_id(0)
    pid_bh_kv = tl.program_id(1)
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv
    i_t_loc = t_offset + pid_local
    i_t = t_base + i_t_loc
    rightmost_idx = i_t // layer_power
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE
    prev_sel_base = (i_b.to(tl.int64) * T * H_kv * TOP_K + i_t_loc.to(tl.int64) * H_kv * TOP_K
                     + i_h_kv.to(tl.int64) * TOP_K)
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))
    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    sbase = (pid_bh_kv.to(tl.int64) * n_prog_t + pid_local) * MAX_CAND
    mbase = (pid_bh_kv.to(tl.int64) * n_prog_t + pid_local) * TOP_K
    idx_mask = (1 << LOG_N) - 1
    o_bc = tl.arange(0, BC).to(tl.int32)
    o_nt = tl.arange(0, NTILE).to(tl.int32)
    row2 = o_nt[:, None]
    slot2 = row2 * BC + o_bc[None, :]                                        # [NTILE, BC]
    live2 = slot2 < num_candidates
    v = tl.load(raw + sbase + slot2.to(tl.int64), mask=live2, other=NEG_INF)
    par2 = slot2 // COMPRESSION_RATE
    m_par = tl.load(pmax + mbase + par2.to(tl.int64), mask=live2, other=NEG_INF)   # parent-local max
    v = _replay_chain(v, m_par, live2, slot2, num_candidates,
                      (num_valid_parents + PARENTS_PER_BATCH - 1) // PARENTS_PER_BATCH, NTILE, LOG_N, NEG_INF)
    vals = v
    vi = vals.to(tl.int32, bitcast=True)
    enc_idx = tl.where(vals >= 0, ~slot2, slot2) & idx_mask
    enc = (vi & ~idx_mask) | enc_idx
    us = (enc ^ ((enc >> 31) | SIGN)) ^ SIGN
    taus = _rank_select_early(us, TOP_K)
    keep = (us >= taus) & (vals > SCORE_VALID_THRESHOLD) & (slot2 < num_candidates)
    parent_idx_f = tl.load(prev_selected_parents + prev_sel_base + par2.to(tl.int64), mask=keep, other=-1).to(tl.int32)
    node = tl.where(keep & (parent_idx_f >= 0), parent_idx_f * COMPRESSION_RATE + slot2 % COMPRESSION_RATE, -1)
    ok = (node >= 0).to(tl.int32)
    row_cnt = tl.sum(ok, axis=1)
    rk = (tl.cumsum(row_cnt, axis=0) - row_cnt)[:, None] + tl.cumsum(ok, axis=1) - ok
    tl.store(next_selected_parents + prev_sel_base + rk.to(tl.int64), node, mask=(ok > 0) & (rk < TOP_K))


@triton.jit
def mid_select_union_kernel(
    q, krot, ulist, umask, ucount, posq, cos_cache, sin_cache, raw, pmax,
    t_offset, t_base, n_prog_t, T, N_layer, UMAX,
    layer_power: tl.constexpr, H: tl.constexpr, H_kv: tl.constexpr, G: tl.constexpr, BQ: tl.constexpr,
    K: tl.constexpr, CR: tl.constexpr, TOP_K: tl.constexpr, NEG_INF: tl.constexpr, scale, MMA_STATE: tl.constexpr,
    RES_OFF,
):
    MAX_CAND: tl.constexpr = TOP_K * CR
    R: tl.constexpr = BQ * G
    i_blk = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h = pid % H_kv
    r = tl.arange(0, R)
    g = r % G
    qi = r // G
    pid_local = i_blk * BQ + qi
    row_ok = pid_local < n_prog_t
    pid_local = tl.minimum(pid_local, n_prog_t - 1)
    t_loc = t_offset + pid_local
    t_glob = t_base + t_loc
    rm_idx = t_glob // layer_power
    rp = rm_idx // CR
    rc = rm_idx % CR
    pq = tl.load(posq + (i_b * T + t_loc).to(tl.int64) * H_kv + i_h)
    jq = tl.arange(0, BQ)
    pl_q = tl.minimum(i_blk * BQ + jq, n_prog_t - 1)
    q_ok = (i_blk * BQ + jq) < n_prog_t
    rbase = (pid.to(tl.int64) * n_prog_t + pl_q) * MAX_CAND                # [BQ] scratch row of each query
    mbase = (pid.to(tl.int64) * n_prog_t + pl_q) * TOP_K                   # [BQ] per-parent max row

    o_kh = tl.arange(0, K // 2).to(tl.int64)
    head = (i_h * G + g).to(tl.int64)
    qp = q + ((i_b * T + t_loc).to(tl.int64) * H + head) * K
    if MMA_STATE:
        eye = (tl.arange(0, K // 2)[:, None] == tl.arange(0, K // 2)[None, :]).to(q.dtype.element_ty)
        q_lo = tl.dot(tl.load(qp[:, None] + o_kh[None, :], mask=row_ok[:, None], other=0.0), eye)
        q_hi = tl.dot(tl.load(qp[:, None] + (K // 2) + o_kh[None, :], mask=row_ok[:, None], other=0.0), eye)
    else:
        q_lo = tl.load(qp[:, None] + o_kh[None, :], mask=row_ok[:, None], other=0.0).to(tl.float32)
        q_hi = tl.load(qp[:, None] + (K // 2) + o_kh[None, :], mask=row_ok[:, None], other=0.0).to(tl.float32)
    cq = tl.load(cos_cache + pq.to(tl.int64)[:, None] * (K // 2) + o_kh[None, :])
    sq = tl.load(sin_cache + pq.to(tl.int64)[:, None] * (K // 2) + o_kh[None, :])
    a_lo = (q_lo * cq - q_hi * sq) * scale
    a_hi = (q_lo * sq + q_hi * cq) * scale
    c16 = tl.load(cos_cache + CR * (K // 2) + o_kh)
    s16 = tl.load(sin_cache + CR * (K // 2) + o_kh)

    ptr_q = tl.zeros([BQ], dtype=tl.int32)
    m_q = tl.full([BQ], NEG_INF, dtype=tl.float32)
    ch = tl.arange(0, CR)
    kb = krot + (i_b.to(tl.int64) * N_layer * H_kv + i_h) * K
    ub = (i_b * tl.num_programs(0) + i_blk).to(tl.int64) * H_kv + i_h
    n_iter = tl.load(ucount + ub)
    lp = ulist + ub * UMAX
    lm = umask + ub * UMAX
    for it in range(n_iter):
        p = tl.load(lp + it)
        m = tl.load(lm + it)
        mem = row_ok & (((m >> qi) & 1) != 0)
        mem_q = q_ok & (((m >> jq) & 1) != 0)                                      # [BQ]
        tok = p * CR + ch
        tok_ok = tok < N_layer
        tok_s = tl.minimum(tok, N_layer - 1).to(tl.int64)
        valid = mem[:, None] & tok_ok[None, :] & ((p != rp)[:, None] | (ch[None, :] <= rc[:, None]))
        k_lo = tl.load(kb + tok_s[:, None] * (H_kv * K) + o_kh[None, :])
        k_hi = tl.load(kb + tok_s[:, None] * (H_kv * K) + (K // 2) + o_kh[None, :])
        kl_r = tl.load(kb + RES_OFF + tok_s[:, None] * (H_kv * K) + o_kh[None, :])
        kh_r = tl.load(kb + RES_OFF + tok_s[:, None] * (H_kv * K) + (K // 2) + o_kh[None, :])
        al_h = a_lo.to(tl.float16)
        ah_h = a_hi.to(tl.float16)
        al_r = (a_lo - al_h.to(tl.float32)).to(tl.float16)
        ah_r = (a_hi - ah_h.to(tl.float32)).to(tl.float16)
        s = (tl.dot(al_h, tl.trans(k_lo)) + tl.dot(ah_h, tl.trans(k_hi))
             + tl.dot(al_h, tl.trans(kl_r)) + tl.dot(ah_h, tl.trans(kh_r))
             + tl.dot(al_r, tl.trans(k_lo)) + tl.dot(ah_r, tl.trans(k_hi)))
        s = tl.where(valid, s, NEG_INF)
        s3 = tl.reshape(s, [BQ, G, CR])
        v3 = tl.reshape(valid, [BQ, G, CR])
        tile_m = tl.max(tl.max(s3, axis=2), axis=1)                                   # [BQ] local max
        new_m = tile_m
        e = tl.where(v3, tl.exp(s3 - tile_m[:, None, None]), 0.0)
        imp = tl.sum(e, axis=1)                                                        # [BQ, CR]
        vq = tl.max(v3.to(tl.int32), axis=1) > 0
        imp = tl.where(vq, imp, NEG_INF)
        tl.store(raw + rbase[:, None] + (ptr_q[:, None] * CR + ch[None, :]).to(tl.int64), imp, mask=mem_q[:, None])
        tl.store(pmax + mbase + ptr_q.to(tl.int64), new_m, mask=mem_q)
        m_q = new_m
        ptr_q = ptr_q + mem_q.to(tl.int32)
        b_lo = a_lo * c16[None, :] + a_hi * s16[None, :]
        b_hi = a_hi * c16[None, :] - a_lo * s16[None, :]
        a_lo = tl.where(mem[:, None], b_lo, a_lo)
        a_hi = tl.where(mem[:, None], b_hi, a_hi)


def fast_mid_select_union(q, k_layer, sel, cos_cache, sin_cache, next_sel, *, t_offset, t_base, n_prog_t, N_layer,
                          layer_power, CR, LOG_N, score_valid_threshold, neg_inf, scale, ul, krot=None, posq=None,
                          work=None):
    """Drop-in for one prefill recsa_mid_select_kernel chunk launch; ul is the prebuilt union_lists of the chunk."""
    B, T, H, K = q.shape
    Hkv = k_layer.shape[2]
    TOP_K = sel.shape[-1]
    if krot is None or (krot.dtype, krot.dim()) != (torch.float16, 5):
        krot = rotate_keys(k_layer, cos_cache, sin_cache, CR, split=True)
    if posq is None:
        posq = query_positions(sel, t_base, CR, layer_power)
    need = B * Hkv * n_prog_t
    if work is None or work[0].numel() < need * TOP_K * CR:
        work = (torch.empty(need * TOP_K * CR, dtype=torch.float32, device=q.device),
                torch.empty(need * TOP_K, dtype=torch.float32, device=q.device))
    raw, pmax = work
    ulist, umask, ucount, BQ = ul
    if _mid_a_cuda_ok(BQ, H // Hkv, K, CR, TOP_K, q, krot, ulist, posq, cos_cache, sin_cache):
        _TB.mid_score(q, krot, ulist, umask, ucount, posq, cos_cache, sin_cache, raw, pmax, t_offset=t_offset,
                      t_base=t_base, n_prog_t=n_prog_t, N_layer=N_layer, layer_power=layer_power, scale=scale,
                      neg_inf=neg_inf)
    else:
        mid_select_union_kernel[(triton.cdiv(n_prog_t, BQ), B * Hkv)](
            q, krot, ulist, umask, ucount, posq, cos_cache, sin_cache, raw, pmax,
            t_offset, t_base, n_prog_t, T, N_layer, ulist.shape[-1],
            layer_power=layer_power, H=H, H_kv=Hkv, G=H // Hkv, BQ=BQ, K=K, CR=CR, TOP_K=TOP_K, NEG_INF=neg_inf,
            scale=scale, MMA_STATE=int(q.dtype in (torch.bfloat16, torch.float16)), RES_OFF=krot[0].numel(),
            num_warps=4 if BQ >= 16 else 1, num_stages=3)
    if TOP_K == 256 and CR == 16 and _top_b_cuda_ok(64, TOP_K):
        _TB.mid_finalize(raw, pmax, sel, next_sel, t_offset=t_offset, t_base=t_base, n_prog_t=n_prog_t, T=T,
                         B_Hkv=B * Hkv, layer_power=layer_power, H_kv=Hkv, LOG_N=LOG_N, svt=score_valid_threshold,
                         neg_inf=neg_inf)
    else:
        mid_select_finalize_kernel[(n_prog_t, B * Hkv)](
            raw, pmax, sel, next_sel, t_offset, t_base, n_prog_t, T,
            layer_power=layer_power, H_kv=Hkv, COMPRESSION_RATE=CR, TOP_K=TOP_K, LOG_N=LOG_N,
            SCORE_VALID_THRESHOLD=score_valid_threshold, NEG_INF=neg_inf, num_warps=4)
    return krot, posq, work


@triton.jit
def rotate_keys_by_pos_kernel(k_in, k_out, cos_cache, sin_cache, NTOT, N_layer, H_kv: tl.constexpr, K: tl.constexpr,
                              BLOCK_N: tl.constexpr, SPLIT: tl.constexpr = 0, RES_OFF=0):
    """k_out[b, n, h] = R(n) k_in[b, n, h] (position = node index): fp32 math, stored fp16."""
    pid = tl.program_id(0)
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = n < NTOT
    o_kh = tl.arange(0, K // 2).to(tl.int64)
    pos = (n % N_layer).to(tl.int64)
    cs = tl.load(cos_cache + pos[:, None] * (K // 2) + o_kh[None, :], mask=ok[:, None], other=1.0)
    sn = tl.load(sin_cache + pos[:, None] * (K // 2) + o_kh[None, :], mask=ok[:, None], other=0.0)
    for h in range(H_kv):
        base = (n.to(tl.int64) * H_kv + h) * K
        lo = tl.load(k_in + base[:, None] + o_kh[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        hi = tl.load(k_in + base[:, None] + (K // 2) + o_kh[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        r_lo = lo * cs - hi * sn
        r_hi = lo * sn + hi * cs
        tl.store(k_out + base[:, None] + o_kh[None, :], r_lo.to(k_out.dtype.element_ty), mask=ok[:, None])
        tl.store(k_out + base[:, None] + (K // 2) + o_kh[None, :], r_hi.to(k_out.dtype.element_ty), mask=ok[:, None])
        if SPLIT:
            tl.store(k_out + RES_OFF + base[:, None] + o_kh[None, :],
                     (r_lo - r_lo.to(tl.float16).to(tl.float32)).to(tl.float16), mask=ok[:, None])
            tl.store(k_out + RES_OFF + base[:, None] + (K // 2) + o_kh[None, :],
                     (r_hi - r_hi.to(tl.float16).to(tl.float32)).to(tl.float16), mask=ok[:, None])


@triton.jit
def top_select_score_kernel(
    q, krot, raw, pmax, cos_cache, sin_cache,
    t_base, q0, T, QC, N_layer, NP, NT,
    layer_power: tl.constexpr, H: tl.constexpr, H_kv: tl.constexpr, G: tl.constexpr, QB: tl.constexpr,
    K: tl.constexpr, TILE_N: tl.constexpr, NEG_INF: tl.constexpr, scale, RES_OFF, ascr,
):
    R: tl.constexpr = QB * G
    i_blk = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h = pid % H_kv
    r = tl.arange(0, R)
    g = r % G
    lq = i_blk * QB + r // G                              # chunk-local query of the row
    row_ok = (lq < QC) & (q0 + lq < T)
    t_loc = tl.minimum(q0 + lq, T - 1)
    pos_q = (t_base + t_loc) // layer_power               # rightmost node == query RoPE position
    ncand = pos_q + 1
    o_kh = tl.arange(0, K // 2).to(tl.int64)
    head = (i_h * G + g).to(tl.int64)
    qp = q + ((i_b * T + t_loc).to(tl.int64) * H + head) * K
    q_lo = tl.load(qp[:, None] + o_kh[None, :], mask=row_ok[:, None], other=0.0).to(tl.float32)
    q_hi = tl.load(qp[:, None] + (K // 2) + o_kh[None, :], mask=row_ok[:, None], other=0.0).to(tl.float32)
    cq = tl.load(cos_cache + pos_q.to(tl.int64)[:, None] * (K // 2) + o_kh[None, :])
    sq = tl.load(sin_cache + pos_q.to(tl.int64)[:, None] * (K // 2) + o_kh[None, :])
    a_lo32 = (q_lo * cq - q_hi * sq) * scale
    a_hi32 = (q_lo * sq + q_hi * cq) * scale
    a_lo = a_lo32.to(tl.float16)
    a_hi = a_hi32.to(tl.float16)
    a_lo_r = (a_lo32 - a_lo.to(tl.float32)).to(tl.float16)       # fp16 residuals of the fp32 rotated q
    a_hi_r = (a_hi32 - a_hi.to(tl.float32)).to(tl.float16)
    # stage [hi | residual] rows once; reload as full-width [R, K] operands
    o_k = tl.arange(0, K).to(tl.int64)
    sp = ascr + ((pid.to(tl.int64) * tl.num_programs(0) + i_blk) * R + r.to(tl.int64))[:, None] * (2 * K)
    tl.store(sp + o_kh[None, :], a_lo)
    tl.store(sp + (K // 2) + o_kh[None, :], a_hi)
    tl.store(sp + K + o_kh[None, :], a_lo_r)
    tl.store(sp + K + (K // 2) + o_kh[None, :], a_hi_r)
    tl.debug_barrier()
    a_full = tl.load(sp + o_k[None, :])
    ar_full = tl.load(sp + K + o_k[None, :])

    jq = tl.arange(0, QB)
    lqq = i_blk * QB + jq
    q_ok = (lqq < QC) & (q0 + lqq < T)
    rbase = (pid.to(tl.int64) * QC + lqq) * NP                                 # [QB] raw row
    mbase = (pid.to(tl.int64) * QC + lqq) * NT                                 # [QB] pmax row
    kb = krot + (i_b.to(tl.int64) * N_layer * H_kv + i_h) * K
    nmax = (t_base + tl.minimum(q0 + i_blk * QB + QB - 1, T - 1)) // layer_power + 1
    o_n = tl.arange(0, TILE_N)
    for it in range(tl.cdiv(nmax, TILE_N)):
        node = it * TILE_N + o_n
        nd_s = tl.minimum(node, N_layer - 1).to(tl.int64)
        k_full = tl.load(kb + nd_s[:, None] * (H_kv * K) + o_k[None, :])
        kr_full = tl.load(kb + RES_OFF + nd_s[:, None] * (H_kv * K) + o_k[None, :])
        s = (tl.dot(a_full, tl.trans(k_full)) + tl.dot(a_full, tl.trans(kr_full))
             + tl.dot(ar_full, tl.trans(k_full)))
        valid = row_ok[:, None] & (node[None, :] < ncand[:, None])
        s = tl.where(valid, s, NEG_INF)
        s3 = tl.reshape(s, [QB, G, TILE_N])
        v3 = tl.reshape(valid, [QB, G, TILE_N])
        m_t = tl.max(tl.max(s3, axis=2), axis=1)                               # [QB]
        e = tl.where(v3, tl.exp(s3 - m_t[:, None, None]), 0.0)
        imp = tl.sum(e, axis=1)                                                # [QB, TILE_N]
        tl.store(raw + rbase[:, None] + node[None, :].to(tl.int64), imp, mask=q_ok[:, None] & (node[None, :] < NP))
        tl.store(pmax + mbase + it, m_t, mask=q_ok)


@triton.jit
def top_select_finalize_kernel(
    raw, pmax, next_sel, bitmask,
    t_base, q0, T, QC, NP, NT, NWORDS, lq0,
    layer_power: tl.constexpr, H_kv: tl.constexpr, TOP_K: tl.constexpr, LOG_N: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr, NEG_INF: tl.constexpr, TILE_N: tl.constexpr, NR: tl.constexpr,
):
    SIGN: tl.constexpr = -2147483648
    BC: tl.constexpr = 256
    lq = lq0 + tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h = pid % H_kv
    t_loc = q0 + lq
    ncand = (t_base + t_loc) // layer_power + 1
    rrow = (pid.to(tl.int64) * QC + lq)
    o_r = tl.arange(0, NR)
    o_c = tl.arange(0, BC)
    node2 = o_r[:, None] * BC + o_c[None, :]                                   # [NR, BC]
    live = node2 < ncand
    v = tl.load(raw + rrow * NP + node2.to(tl.int64), mask=live, other=NEG_INF)
    m_par = tl.load(pmax + rrow * NT + (node2 // TILE_N).to(tl.int64), mask=live, other=NEG_INF)  # tile-local max
    v = _replay_chain(v, m_par, live, node2, ncand, (ncand + BC - 1) // BC, NR, LOG_N, NEG_INF)
    idx_mask = (1 << LOG_N) - 1
    vi = v.to(tl.int32, bitcast=True)
    enc_idx = tl.where(v >= 0, ~node2, node2) & idx_mask
    enc = (vi & ~idx_mask) | enc_idx
    us = (enc ^ ((enc >> 31) | SIGN)) ^ SIGN
    if ncand > TOP_K:
        taus = _rank_select_early(us, TOP_K)
    else:                                   # every live candidate is kept
        taus = tl.full([1], SIGN, dtype=tl.int32)
    keep = (us >= taus) & (v > SCORE_VALID_THRESHOLD) & live
    ok = keep.to(tl.int32)
    row_cnt = tl.sum(ok, axis=1)
    rk = (tl.cumsum(row_cnt, axis=0) - row_cnt)[:, None] + tl.cumsum(ok, axis=1) - ok
    obase = ((i_b * T + t_loc).to(tl.int64) * H_kv + i_h) * TOP_K
    tl.store(next_sel + obase + rk.to(tl.int64), node2, mask=keep & (rk < TOP_K))
    # selected bitmask words (the top accumulate excludes these nodes)
    kb3 = tl.reshape(ok, [NR, BC // 32, 32])
    words = tl.sum(kb3 << tl.arange(0, 32)[None, None, :], axis=2)            # [NR, BC // 32]
    widx = o_r[:, None] * (BC // 32) + tl.arange(0, BC // 32)[None, :]
    wbase = ((i_b * T + t_loc).to(tl.int64) * H_kv + i_h) * NWORDS
    tl.store(bitmask + wbase + widx.to(tl.int64), words, mask=widx < NWORDS)


FAST_TOP_B_CUDA = 1
FAST_MIDA_CUDA = 1
FAST_TOPA_CUDA = 1
_TB = None


def _top_b_cuda_ok(TILE_N, TOP_K):
    global _TB, FAST_TOP_B_CUDA
    if not FAST_TOP_B_CUDA or TILE_N != 64 or torch.cuda.get_device_capability() != (9, 0):
        return False
    if _TB is None:
        try:
            from recsa import topsel_cuda
            topsel_cuda.build()
            _TB = topsel_cuda
        except Exception as ex:      # no nvcc / driver: keep the Triton kernel
            from recsa.leaf_cuda import cuda_fallback
            cuda_fallback("top_finalize", ex)           # raises under strict mode (eval runs)
            FAST_TOP_B_CUDA = 0
            print(f"[recsa] CUDA select kernel B: build failed, using the Triton kernel ({ex})", flush=True)
            return False
    return True


def _mid_a_cuda_ok(BQ, G, K, CR, TOP_K, q, krot, ulist, posq, cos_cache, sin_cache):
    global FAST_MIDA_CUDA
    if not (FAST_MIDA_CUDA and BQ == 16 and G == 4 and K == 128 and CR == 16 and TOP_K == 256
            and q.dtype == torch.bfloat16 and q.is_contiguous() and krot.dim() == 5 and krot.is_contiguous()
            and krot.data_ptr() % 16 == 0 and ulist.is_contiguous() and posq.is_contiguous()
            and cos_cache.is_contiguous() and sin_cache.is_contiguous() and cos_cache.dtype == torch.float32
            and _top_b_cuda_ok(64, TOP_K)):
        return False
    try:
        _TB._mmodule()
    except Exception as ex:          # no nvcc / driver: keep the Triton kernel
        from recsa.leaf_cuda import cuda_fallback
        cuda_fallback("mid_score", ex)                  # raises under strict mode (eval runs)
        FAST_MIDA_CUDA = 0
        print(f"[recsa] CUDA mid kernel A: build failed, using the Triton kernel ({ex})", flush=True)
        return False
    return True


FAST_TOP_TILE_N = 64
FAST_TOP_QB = 16


def _top_a_cuda_ok(TILE_N, G, K, q, cos_cache):
    global FAST_TOPA_CUDA
    if not (FAST_TOPA_CUDA and TILE_N == 64 and FAST_TOP_QB == 16 and G == 4 and K == 128
            and q.dtype == torch.bfloat16 and q.is_contiguous() and cos_cache.dtype == torch.float32
            and cos_cache.is_contiguous() and torch.cuda.get_device_capability(q.device) == (9, 0)):
        return False
    try:
        from recsa import topsel_cuda
        topsel_cuda._amodule()
    except Exception as ex:  # noqa: BLE001
        from recsa.leaf_cuda import cuda_fallback
        cuda_fallback("top_score", ex)                  # raises under strict mode (eval runs)
        FAST_TOPA_CUDA = 0
        print(f"[recsa] CUDA top kernel A: build failed, using the Triton kernel ({ex})", flush=True)
        return False
    return True
FAST_TOP_MAX_NODES = 4096


def fast_top_select(q, k_layer, cos_cache, sin_cache, next_sel, bitmask, *, t_base, layer_power, N_layer, LOG_N,
                    score_valid_threshold, neg_inf, scale, work_bytes=1 << 30):
    """Drop-in for the prefill recsa_top_select_kernel launch (next_sel pre-filled with -1; bitmask words written)."""
    if N_layer > FAST_TOP_MAX_NODES:
        return None
    B, T, H, K = q.shape
    Hkv = k_layer.shape[2]
    TOP_K = next_sel.shape[-1]
    NWORDS = bitmask.shape[-1]
    TILE_N = FAST_TOP_TILE_N
    NP = triton.cdiv(N_layer, TILE_N) * TILE_N
    NT = NP // TILE_N
    NR = max(1, triton.next_power_of_2(NP) // 256)
    k_layer = k_layer.contiguous()
    krot = torch.empty((2,) + tuple(k_layer.shape), dtype=torch.float16, device=k_layer.device)
    rotate_keys_by_pos_kernel[(triton.cdiv(B * N_layer, 64),)](k_layer, krot, cos_cache, sin_cache, B * N_layer, N_layer,
                                                                H_kv=Hkv, K=K, BLOCK_N=64, SPLIT=1,
                                                                RES_OFF=k_layer.numel(), num_warps=4)
    QC = max(1024, min(T, work_bytes // (4 * B * Hkv * NP)))
    raw = torch.empty(B * Hkv * QC * NP, dtype=torch.float32, device=q.device)
    pmax = torch.empty(B * Hkv * QC * NT, dtype=torch.float32, device=q.device)
    use_cuda_a = _top_a_cuda_ok(TILE_N, H // Hkv, K, q, cos_cache)
    ascr = (torch.empty(B * Hkv * triton.cdiv(QC, FAST_TOP_QB) * FAST_TOP_QB * (H // Hkv) * 2 * K,
                        dtype=torch.float16, device=q.device) if not use_cuda_a else None)
    use_cuda_b = _top_b_cuda_ok(TILE_N, TOP_K)
    if use_cuda_a:
        from recsa import topsel_cuda
        kmap = topsel_cuda.krot_map(krot)
    for q0 in range(0, T, QC):
        qc = min(QC, T - q0)
        if use_cuda_a:
            topsel_cuda.top_score(q, krot, raw, pmax, cos_cache, sin_cache, t_base=t_base, q0=q0, qc=qc, T=T, QC=QC,
                                  N_layer=N_layer, NP=NP, NT=NT, layer_power=layer_power, scale=scale, neg_inf=neg_inf,
                                  kmap=kmap)
        else:   # K = 128 dots, one pipeline stage
            top_select_score_kernel[(triton.cdiv(qc, FAST_TOP_QB), B * Hkv)](
                q, krot, raw, pmax, cos_cache, sin_cache, t_base, q0, T, QC, N_layer, NP, NT,
                layer_power=layer_power, H=H, H_kv=Hkv, G=H // Hkv, QB=FAST_TOP_QB, K=K, TILE_N=TILE_N,
                NEG_INF=neg_inf, scale=scale, RES_OFF=k_layer.numel(), ascr=ascr, num_warps=4, num_stages=1)
        lq = 0
        while lq < qc:
            nr, end = NR, qc
            nc0 = (t_base + q0 + lq) // layer_power + 1
            need = triton.cdiv(nc0, 256)
            if use_cuda_b:
                nr = min(NR, min(x for x in _TB.NR_SET if x >= min(need, 16)))
            else:
                nr = min(NR, max(1, triton.next_power_of_2(need)))
            if nr < NR:
                end = min(qc, nr * 256 * layer_power - t_base - q0)
            if use_cuda_b:
                _TB.top_finalize(raw, pmax, next_sel, bitmask, t_base=t_base, q0=q0, T=T, QC=QC, NP=NP, NT=NT,
                                 NWORDS=NWORDS, lq0=lq, nq=end - lq, nr=nr, B_Hkv=B * Hkv, layer_power=layer_power,
                                 H_kv=Hkv, TOP_K=TOP_K, LOG_N=LOG_N, svt=score_valid_threshold, neg_inf=neg_inf)
            else:
                top_select_finalize_kernel[(end - lq, B * Hkv)](
                    raw, pmax, next_sel, bitmask, t_base, q0, T, QC, NP, NT, NWORDS, lq,
                    layer_power=layer_power, H_kv=Hkv, TOP_K=TOP_K, LOG_N=LOG_N,
                    SCORE_VALID_THRESHOLD=score_valid_threshold, NEG_INF=neg_inf, TILE_N=TILE_N, NR=nr,
                    num_warps=4)
            lq = end
    return krot[0]          # the hi half: fp16 rotated keys


def rotate_top_keys(k_layer, cos_cache, sin_cache):
    """Top-level keys rotated by node position (fp32 math, fp16 storage), [B, N, H_kv, K]."""
    B, N, Hkv, K = k_layer.shape
    k_layer = k_layer.contiguous()
    krot = torch.empty(k_layer.shape, dtype=torch.float16, device=k_layer.device)
    rotate_keys_by_pos_kernel[(triton.cdiv(B * N, 64),)](k_layer, krot, cos_cache, sin_cache, B * N, N,
                                                         H_kv=Hkv, K=K, BLOCK_N=64, num_warps=4)
    return krot
