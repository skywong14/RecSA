"""Union-walk block kernels for the prefill leaf / mid accumulations."""

import torch
import triton
import triton.language as tl


@triton.jit
def rotate_keys_by_child_kernel(k_in, k_out, cos_cache, sin_cache, N, H_kv: tl.constexpr, K: tl.constexpr,
                                CR: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT: tl.constexpr = 0, RES_OFF=0):
    """k_out[n, h] = R(n % CR) k_in[n, h]: fp32 math, stored fp16. k_in / k_out [B*N, H_kv, K] contiguous."""
    pid = tl.program_id(0)
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = n < N
    o_kh = tl.arange(0, K // 2).to(tl.int64)
    c = (n % CR).to(tl.int64)
    cs = tl.load(cos_cache + c[:, None] * (K // 2) + o_kh[None, :])
    sn = tl.load(sin_cache + c[:, None] * (K // 2) + o_kh[None, :])
    for h in range(H_kv):
        base = (n.to(tl.int64) * H_kv + h) * K
        lo = tl.load(k_in + base[:, None] + o_kh[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        hi = tl.load(k_in + base[:, None] + (K // 2) + o_kh[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        r_lo = lo * cs - hi * sn
        r_hi = lo * sn + hi * cs
        tl.store(k_out + base[:, None] + o_kh[None, :], r_lo.to(k_out.dtype.element_ty), mask=ok[:, None])
        tl.store(k_out + base[:, None] + (K // 2) + o_kh[None, :], r_hi.to(k_out.dtype.element_ty), mask=ok[:, None])
        if SPLIT:   # fp16 residual of the fp32 rotated key at RES_OFF
            tl.store(k_out + RES_OFF + base[:, None] + o_kh[None, :],
                     (r_lo - r_lo.to(tl.float16).to(tl.float32)).to(tl.float16), mask=ok[:, None])
            tl.store(k_out + RES_OFF + base[:, None] + (K // 2) + o_kh[None, :],
                     (r_hi - r_hi.to(tl.float16).to(tl.float32)).to(tl.float16), mask=ok[:, None])


def rotate_keys(k, cos_cache, sin_cache, CR, dtype=torch.float16, split=False):
    """R(node % CR) k for a level's keys [B, N, H_kv, K] -> dtype (fp16; fp32 for split-precision selection)."""
    B, N, Hkv, K = k.shape
    k = k.contiguous()
    if split:
        out = torch.empty((2,) + tuple(k.shape), dtype=torch.float16, device=k.device)
    else:
        out = torch.empty(k.shape, dtype=dtype, device=k.device)
    BLOCK_N = 64
    rotate_keys_by_child_kernel[(triton.cdiv(B * N, BLOCK_N),)](k, out, cos_cache, sin_cache, B * N, H_kv=Hkv, K=K,
                                                                CR=CR, BLOCK_N=BLOCK_N, SPLIT=int(split),
                                                                RES_OFF=k.numel() if split else 0, num_warps=4)
    return out


def query_positions(sel, t_base, CR, layer_power):
    """Query position in candidate order, (nvalid - 1) * CR + rightmost child, as int32 [B, T, H_kv]."""
    B, T, Hkv, _ = sel.shape
    nv = (sel >= 0).sum(-1, dtype=torch.int32)
    t = torch.arange(T, device=sel.device, dtype=torch.int32) + t_base
    return ((nv - 1) * CR + ((t // layer_power) % CR)[None, :, None]).to(torch.int32).contiguous()


UNION_BIGV = 1 << 30
UNION_LEAF_DENSE_NODES = 65536


def union_lists_torch(sel, BQ, umax=None):
    """Per block of BQ queries and kv-head: the sorted distinct selected parents, a BQ-bit mask of the queries that
    selected each, and their count. sel int32 [B, T, H_kv, TK] (-1 = empty) -> ulist, umask [B, NB, H_kv, U],
    ucount [B, NB, H_kv]."""
    B, T, Hkv, TK = sel.shape
    NB = (T + BQ - 1) // BQ
    pad = NB * BQ - T
    if pad:
        sel = torch.cat([sel, sel.new_full((B, pad, Hkv, TK), -1)], 1)
    s = sel.reshape(B, NB, BQ, Hkv, TK).permute(0, 1, 3, 2, 4)
    qi = torch.arange(BQ, device=sel.device, dtype=torch.int32).view(1, 1, 1, BQ, 1)
    key = torch.where(s >= 0, s * BQ + qi, torch.full_like(s, UNION_BIGV)).reshape(B, NB, Hkv, BQ * TK)
    key = key.sort(-1).values
    valid = key < UNION_BIGV
    par = torch.where(valid, key // BQ, torch.full_like(key, UNION_BIGV))
    bit = torch.where(valid, torch.ones_like(key) << (key % BQ), torch.zeros_like(key))
    first = torch.ones_like(key, dtype=torch.bool)
    first[..., 1:] = par[..., 1:] != par[..., :-1]
    seg = first.cumsum(-1) - 1
    ulist = torch.full_like(key, UNION_BIGV).scatter_(-1, seg, par)
    umask = torch.zeros_like(key).scatter_add_(-1, seg, bit)
    ucount = (first & valid).sum(-1, dtype=torch.int32)
    if umax is not None and umax < ulist.shape[-1]:
        ulist, umask = ulist[..., :umax], umask[..., :umax]
    return ulist.contiguous(), umask.contiguous(), ucount.contiguous()


def union_xmask_torch(next_sel, ulist, BQ, CR):
    """Exclusion masks for mid_acc_union_kernel: per union entry j and query qi, the CR-bit mask of the children of
    ulist[j] that the query selected at this level. int32 [B, NB, H_kv, U * BQ]."""
    B, T, Hkv, TK = next_sel.shape
    NB, U = ulist.shape[1], ulist.shape[-1]
    pad = NB * BQ - T
    if pad:
        next_sel = torch.cat([next_sel, next_sel.new_full((B, pad, Hkv, TK), -1)], 1)
    s = next_sel.reshape(B, NB, BQ, Hkv, TK).permute(0, 1, 3, 2, 4)                  # [B, NB, Hkv, BQ, TK]
    valid = s >= 0
    par = torch.where(valid, s // CR, torch.full_like(s, UNION_BIGV)).reshape(B * NB * Hkv, BQ * TK)
    j = torch.searchsorted(ulist.reshape(B * NB * Hkv, U), par.contiguous()).reshape(B, NB, Hkv, BQ, TK)
    qi = torch.arange(BQ, device=s.device).view(1, 1, 1, BQ, 1)
    idx = torch.where(valid, j.clamp_max(U - 1) * BQ + qi, torch.zeros_like(j))
    bit = torch.where(valid, torch.ones_like(s) << (s % CR), torch.zeros_like(s))
    xm = torch.zeros((B, NB, Hkv, U * BQ), dtype=torch.int32, device=s.device)
    return xm.scatter_add_(-1, idx.reshape(B, NB, Hkv, BQ * TK), bit.reshape(B, NB, Hkv, BQ * TK))


@triton.jit
def leaf_union_kernel(
    q, krot, layer_v, ulist, umask, ucount, posq, cos_cache, sin_cache,
    out_max, out_sum, out_o,
    t_base, T, N_layer, UMAX,
    H: tl.constexpr, H_kv: tl.constexpr, G: tl.constexpr, BQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    CR: tl.constexpr, NEG_INF: tl.constexpr, scale, MMA_STATE: tl.constexpr,
):
    R: tl.constexpr = BQ * G
    i_blk = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h = pid % H_kv
    r = tl.arange(0, R)
    g = r % G
    qi = r // G
    t_loc = i_blk * BQ + qi
    row_ok = t_loc < T
    t_loc = tl.minimum(t_loc, T - 1)
    t_glob = t_base + t_loc
    rc = t_glob % CR                                  # leaf: layer_power == 1
    rp = t_glob // CR
    pq = tl.load(posq + (i_b * T + t_loc).to(tl.int64) * H_kv + i_h)

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
    c16 = tl.load(cos_cache + CR * (K // 2) + o_kh)   # R(CR*theta): the cache row at position CR
    s16 = tl.load(sin_cache + CR * (K // 2) + o_kh)

    cur_max = tl.full([R], NEG_INF, dtype=tl.float32)
    cur_sum = tl.zeros([R], dtype=tl.float32)
    cur_out = tl.zeros([R, V], dtype=tl.float32)
    ch = tl.arange(0, CR)
    o_v = tl.arange(0, V).to(tl.int64)
    kb = krot + (i_b.to(tl.int64) * N_layer * H_kv + i_h) * K
    vb = layer_v + (i_b.to(tl.int64) * N_layer * H_kv + i_h) * V
    ub = (i_b * tl.num_programs(0) + i_blk).to(tl.int64) * H_kv + i_h
    n_iter = tl.load(ucount + ub)
    lp = ulist + ub * UMAX
    lm = umask + ub * UMAX
    p = tl.load(lp, mask=n_iter > 0, other=0)
    m = tl.load(lm, mask=n_iter > 0, other=0)
    tok_s = tl.minimum(p * CR + ch, N_layer - 1).to(tl.int64)
    k_lo = tl.load(kb + tok_s[:, None] * (H_kv * K) + o_kh[None, :])
    k_hi = tl.load(kb + tok_s[:, None] * (H_kv * K) + (K // 2) + o_kh[None, :])
    s = tl.dot(a_lo.to(tl.float16), tl.trans(k_lo)) + tl.dot(a_hi.to(tl.float16), tl.trans(k_hi))
    for it in range(n_iter):
        mem = row_ok & (((m >> qi) & 1) != 0)
        tok = p * CR + ch
        tok_ok = tok < N_layer
        tok_s = tl.minimum(tok, N_layer - 1).to(tl.int64)
        valid = mem[:, None] & tok_ok[None, :] & ((p != rp)[:, None] | (ch[None, :] <= rc[:, None]))
        v_t = tl.load(vb + tok_s[:, None] * (H_kv * V) + o_v[None, :]).to(tl.float16)
        b_lo = a_lo * c16[None, :] + a_hi * s16[None, :]
        b_hi = a_hi * c16[None, :] - a_lo * s16[None, :]
        a_lo = tl.where(mem[:, None], b_lo, a_lo)
        a_hi = tl.where(mem[:, None], b_hi, a_hi)
        has_n = it + 1 < n_iter
        pn = tl.load(lp + it + 1, mask=has_n, other=0)
        mn = tl.load(lm + it + 1, mask=has_n, other=0)
        tn_s = tl.minimum(pn * CR + ch, N_layer - 1).to(tl.int64)
        kn_lo = tl.load(kb + tn_s[:, None] * (H_kv * K) + o_kh[None, :])
        kn_hi = tl.load(kb + tn_s[:, None] * (H_kv * K) + (K // 2) + o_kh[None, :])
        s_n = tl.dot(a_lo.to(tl.float16), tl.trans(kn_lo)) + tl.dot(a_hi.to(tl.float16), tl.trans(kn_hi))
        s = tl.where(valid, s, float("-inf"))
        new_m = tl.maximum(cur_max, tl.max(s, axis=1))
        alpha = tl.exp(cur_max - new_m)
        probs = tl.exp(s - new_m[:, None])
        cur_sum = cur_sum * alpha + tl.sum(probs, axis=1)
        cur_out = cur_out * alpha[:, None] + tl.dot(probs.to(tl.float16), v_t)
        cur_max = new_m
        s = s_n
        p = pn
        m = mn

    st = (i_b * T + t_loc).to(tl.int64) * H + head
    tl.store(out_max + st, cur_max, mask=row_ok)
    tl.store(out_sum + st, cur_sum, mask=row_ok)
    tl.store(out_o + st[:, None] * V + o_v[None, :], cur_out, mask=row_ok[:, None])


LEAF_CUDA = 1
_LEAF_CUDA_BROKEN = False


def leaf_union_bq(N_layer, G=4):
    """Queries per union block, at least 16 // G so that BQ * G reaches the 16-row MMA minimum."""
    bq = 16 if N_layer <= UNION_LEAF_DENSE_NODES else 4
    return max(bq, 16 // max(1, G))


def leaf_union(q, k_layer, v_layer, sel, cos_cache, sin_cache, out_max, out_sum, out_o, *, t_base, N_layer, scale,
               CR, neg_inf, BQ=None, fuse=None):
    """Leaf union walk. With fuse, the CUDA walk may write the final normalized output directly; returns True when
    it did."""
    B, T, H, K = q.shape
    Hkv = k_layer.shape[2]
    global _LEAF_CUDA_BROKEN
    if LEAF_CUDA and BQ is None and not _LEAF_CUDA_BROKEN:
        from recsa import leaf_cuda as LC
        if LC.supported(q, k_layer, v_layer, CR):
            try:
                LC.build()
            except Exception as ex:      # no nvcc / driver: keep the Triton walk
                LC.cuda_fallback("leaf_union", ex)      # raises under strict mode
                _LEAF_CUDA_BROKEN = True
                print(f"[recsa] CUDA leaf walk: build failed, using the Triton leaf walk ({ex})", flush=True)
            else:
                krot = rotate_keys(k_layer, cos_cache, sin_cache, CR)
                v16 = v_layer if v_layer.dtype == torch.float16 else v_layer.to(torch.float16)
                if LC.NWARP == 4 and sel.shape[-1] == 256 and sel.stride(3) == 1 and sel.stride(2) == 256:
                    # one CTA per block; also returns the query positions
                    ulist, umask, ucount, posq = LC.union_lists16(sel, pad=False, posq_t_base=t_base,
                                                                  n_parents=-(-N_layer // CR))
                else:
                    posq = query_positions(sel, t_base, CR, 1)
                    ulist, umask, ucount = union_lists(sel, 4 * LC.NWARP)
                LC.leaf_union_cuda(q, krot, v16.contiguous(), ulist, umask, ucount, posq, cos_cache, sin_cache,
                                   out_max, out_sum, out_o, t_base=t_base, N_layer=N_layer, scale=scale,
                                   neg_inf=neg_inf, fuse=fuse)
                return fuse is not None
    BQ = BQ or leaf_union_bq(N_layer, H // Hkv)
    krot = rotate_keys(k_layer, cos_cache, sin_cache, CR)
    posq = query_positions(sel, t_base, CR, 1)
    if BQ < 16 and v_layer.dtype == torch.bfloat16:   # fp16 V for the one-warp walk
        v_layer = v_layer.to(torch.float16)
    ulist, umask, ucount = union_lists(sel, BQ)
    leaf_union_kernel[(triton.cdiv(T, BQ), B * Hkv)](
        q, krot, v_layer, ulist, umask, ucount, posq, cos_cache, sin_cache, out_max, out_sum, out_o,
        t_base, T, N_layer, ulist.shape[-1],
        H=H, H_kv=Hkv, G=H // Hkv, BQ=BQ, K=K, V=v_layer.shape[-1], CR=CR, NEG_INF=neg_inf, scale=scale,
        MMA_STATE=int(q.dtype in (torch.bfloat16, torch.float16)), num_warps=4 if BQ >= 16 else 1, num_stages=3)
    return False


UNION_MID_DENSE_NODES = 65536


def mid_union_bq(N_layer, G=4):
    """Queries per union block, at least 16 // G so that BQ * G reaches the 16-row MMA minimum."""
    bq = 16 if N_layer <= UNION_MID_DENSE_NODES else 4
    return max(bq, 16 // max(1, G))


@triton.jit
def mid_acc_union_kernel(
    q, krot, layer_v, ulist, umask, ucount, posq, cos_cache, sin_cache,
    out_max, out_sum, out_o, xmask,
    t_offset, t_base, n_prog_t, T, N_layer, UMAX,
    layer_power: tl.constexpr, H: tl.constexpr, H_kv: tl.constexpr, G: tl.constexpr, BQ: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, CR: tl.constexpr, NEG_INF: tl.constexpr, scale, MMA_STATE: tl.constexpr,
):
    R: tl.constexpr = BQ * G
    i_blk = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h = pid % H_kv
    r = tl.arange(0, R)
    g = r % G
    qi = r // G
    pid_local = i_blk * BQ + qi                       # query index inside this mid chunk
    row_ok = pid_local < n_prog_t
    pid_local = tl.minimum(pid_local, n_prog_t - 1)
    t_loc = t_offset + pid_local
    t_glob = t_base + t_loc
    rm_idx = t_glob // layer_power
    rp = rm_idx // CR
    rc = rm_idx % CR
    pq = tl.load(posq + (i_b * T + t_loc).to(tl.int64) * H_kv + i_h)

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

    cur_max = tl.full([R], NEG_INF, dtype=tl.float32)
    cur_sum = tl.zeros([R], dtype=tl.float32)
    cur_out = tl.zeros([R, V], dtype=tl.float32)
    ch = tl.arange(0, CR)
    o_v = tl.arange(0, V).to(tl.int64)
    kb = krot + (i_b.to(tl.int64) * N_layer * H_kv + i_h) * K
    vb = layer_v + (i_b.to(tl.int64) * N_layer * H_kv + i_h) * V
    ub = (i_b * tl.num_programs(0) + i_blk).to(tl.int64) * H_kv + i_h
    n_iter = tl.load(ucount + ub)
    lp = ulist + ub * UMAX
    lm = umask + ub * UMAX
    p = tl.load(lp, mask=n_iter > 0, other=0)
    m = tl.load(lm, mask=n_iter > 0, other=0)
    tok_s = tl.minimum(p * CR + ch, N_layer - 1).to(tl.int64)
    k_lo = tl.load(kb + tok_s[:, None] * (H_kv * K) + o_kh[None, :])
    k_hi = tl.load(kb + tok_s[:, None] * (H_kv * K) + (K // 2) + o_kh[None, :])
    s = tl.dot(a_lo.to(tl.float16), tl.trans(k_lo)) + tl.dot(a_hi.to(tl.float16), tl.trans(k_hi))
    for it in range(n_iter):
        mem = row_ok & (((m >> qi) & 1) != 0)
        tok = p * CR + ch
        tok_ok = tok < N_layer
        tok_s = tl.minimum(tok, N_layer - 1).to(tl.int64)
        xm = tl.load(xmask + ub * (UMAX * BQ) + it * BQ + qi)
        selected = ((xm[:, None] >> ch[None, :]) & 1) != 0
        valid = (mem[:, None] & tok_ok[None, :] & ((p != rp)[:, None] | (ch[None, :] <= rc[:, None]))
                 & ~selected)
        v_t = tl.load(vb + tok_s[:, None] * (H_kv * V) + o_v[None, :]).to(tl.float16)
        b_lo = a_lo * c16[None, :] + a_hi * s16[None, :]
        b_hi = a_hi * c16[None, :] - a_lo * s16[None, :]
        a_lo = tl.where(mem[:, None], b_lo, a_lo)
        a_hi = tl.where(mem[:, None], b_hi, a_hi)
        has_n = it + 1 < n_iter
        pn = tl.load(lp + it + 1, mask=has_n, other=0)
        mn = tl.load(lm + it + 1, mask=has_n, other=0)
        tn_s = tl.minimum(pn * CR + ch, N_layer - 1).to(tl.int64)
        kn_lo = tl.load(kb + tn_s[:, None] * (H_kv * K) + o_kh[None, :])
        kn_hi = tl.load(kb + tn_s[:, None] * (H_kv * K) + (K // 2) + o_kh[None, :])
        s_n = tl.dot(a_lo.to(tl.float16), tl.trans(kn_lo)) + tl.dot(a_hi.to(tl.float16), tl.trans(kn_hi))
        s = tl.where(valid, s, float("-inf"))
        new_m = tl.maximum(cur_max, tl.max(s, axis=1))
        alpha = tl.exp(cur_max - new_m)
        probs = tl.exp(s - new_m[:, None])
        cur_sum = cur_sum * alpha + tl.sum(probs, axis=1)
        cur_out = cur_out * alpha[:, None] + tl.dot(probs.to(tl.float16), v_t)
        cur_max = new_m
        s = s_n
        p = pn
        m = mn

    st = (i_b * T + t_loc).to(tl.int64) * H + head
    tl.store(out_max + st, cur_max, mask=row_ok)
    tl.store(out_sum + st, cur_sum, mask=row_ok)
    tl.store(out_o + st[:, None] * V + o_v[None, :], cur_out, mask=row_ok[:, None])


def mid_acc_union(q, krot, v_layer, sel, posq, cos_cache, sin_cache, out_max, out_sum, out_o, *, t_offset, t_base,
                  n_prog_t, N_layer, layer_power, CR, neg_inf, scale, ul, xmask):
    """Mid accumulate of the dropped candidates for one chunk. ul comes from union_lists and xmask from union_xmask on
    the chunk's rows."""
    B, T, H, K = q.shape
    Hkv = krot.shape[2]
    ulist, umask, ucount, BQ = ul
    mid_acc_union_kernel[(triton.cdiv(n_prog_t, BQ), B * Hkv)](
        q, krot, v_layer, ulist, umask, ucount, posq, cos_cache, sin_cache, out_max, out_sum, out_o, xmask,
        t_offset, t_base, n_prog_t, T, N_layer, ulist.shape[-1],
        layer_power=layer_power, H=H, H_kv=Hkv, G=H // Hkv, BQ=BQ, K=K, V=v_layer.shape[-1], CR=CR, NEG_INF=neg_inf,
        scale=scale, MMA_STATE=int(q.dtype in (torch.bfloat16, torch.float16)), num_warps=4 if BQ >= 16 else 1,
        num_stages=3)


@triton.jit
def _seg_or(f1, v1, f2, v2):
    """Segmented inclusive OR-scan on (flag, value) pairs; flag marks the start of a segment."""
    return f1 | f2, tl.where(f2 != 0, v2, v1 | v2)


@triton.jit
def union_keys_kernel(sel, keys, T, NB, s_b, s_t, H_kv: tl.constexpr, TK: tl.constexpr, BQ: tl.constexpr,
                      BIG: tl.constexpr):
    """keys[(b*NB + blk)*H_kv + h, qi*TK + j] = sel[b, blk*BQ + qi, h, j] * BQ + qi  (BIG for empty / padded rows)"""
    row = tl.program_id(0)
    h = row % H_kv
    bb = row // H_kv
    blk = bb % NB
    b = bb // NB
    qi = tl.arange(0, BQ)[:, None]
    j = tl.arange(0, TK)[None, :]
    t = blk * BQ + qi
    v = tl.load(sel + b.to(tl.int64) * s_b + t.to(tl.int64) * s_t + h * TK + j, mask=t < T, other=-1)
    k = tl.where(v >= 0, v * BQ + qi, BIG)
    tl.store(keys + row.to(tl.int64) * (BQ * TK) + qi * TK + j, k)


@triton.jit
def union_build_kernel(keys, ulist, umask, ucount, U, NK: tl.constexpr, BQ: tl.constexpr, BIG: tl.constexpr,
                       UP: tl.constexpr):
    """Distinct parents, their BQ-bit query masks and the count from a row of sorted keys (parent * BQ + query); ulist
    is padded with BIG up to U."""
    row = tl.program_id(0)
    i = tl.arange(0, NK)
    kb = keys + row.to(tl.int64) * NK
    k = tl.load(kb + i)
    kp = tl.load(kb + i - 1, mask=i > 0, other=-BQ)
    kn = tl.load(kb + i + 1, mask=i < NK - 1, other=BIG)
    valid = k < BIG
    par = k // BQ
    first = valid & (par != kp // BQ)
    last = valid & ((par != kn // BQ) | (kn >= BIG))
    seg = tl.cumsum(first.to(tl.int32), axis=0) - 1
    bit = tl.where(valid, 1 << (k % BQ), 0)
    _, segor = tl.associative_scan((first.to(tl.int32), bit), 0, _seg_or)
    ub = row.to(tl.int64) * U
    tl.store(ulist + ub + seg, par, mask=first & (seg < U))
    tl.store(umask + ub + seg, segor, mask=last & (seg < U))
    cnt = tl.sum(first.to(tl.int32), axis=0)
    tl.store(ucount + row, cnt)
    jj = tl.arange(0, UP)
    tl.store(ulist + ub + jj, tl.full([UP], BIG, tl.int32), mask=(jj >= cnt) & (jj < U))


@triton.jit
def union_xmask_kernel(next_sel, ulist, ucount, xmask, T, NB, U, s_b, s_t, H_kv: tl.constexpr, TK: tl.constexpr,
                       BQ: tl.constexpr, CR: tl.constexpr, NSTEP: tl.constexpr):
    """xmask[row, j*BQ + qi] |= 1 << (n % CR) for every node n query qi of the block selected (j = index of n // CR in
    the row's union list, by binary search). xmask must be zeroed."""
    row = tl.program_id(0)
    h = row % H_kv
    bb = row // H_kv
    blk = bb % NB
    b = bb // NB
    qi = tl.arange(0, BQ)[:, None]
    j = tl.arange(0, TK)[None, :]
    t = blk * BQ + qi
    n = tl.load(next_sel + b.to(tl.int64) * s_b + t.to(tl.int64) * s_t + h * TK + j, mask=t < T, other=-1)
    ok = n >= 0
    p = n // CR
    cnt = tl.load(ucount + row)
    lb = ulist + row.to(tl.int64) * U
    lo = tl.zeros([BQ, TK], dtype=tl.int32)
    hi = tl.zeros([BQ, TK], dtype=tl.int32) + cnt
    for _ in tl.static_range(NSTEP):
        act = lo < hi
        mid = (lo + hi) // 2
        v = tl.load(lb + mid, mask=act & ok, other=0)
        go_right = v < p
        lo = tl.where(act & go_right, mid + 1, lo)
        hi = tl.where(act & ~go_right, mid, hi)
    tl.atomic_or(xmask + row.to(tl.int64) * (U * BQ) + lo * BQ + qi, 1 << (n % CR), mask=ok & (lo < cnt))


def union_lists_triton(sel, BQ, umax=None):
    B, T, Hkv, TK = sel.shape
    assert sel.stride(3) == 1 and sel.stride(2) == TK, "sel rows must be contiguous per (b, t)"
    NB = (T + BQ - 1) // BQ
    NK = BQ * TK
    rows = B * NB * Hkv
    keys = torch.empty((rows, NK), dtype=torch.int32, device=sel.device)
    union_keys_kernel[(rows,)](sel, keys, T, NB, sel.stride(0), sel.stride(1), H_kv=Hkv, TK=TK, BQ=BQ, BIG=UNION_BIGV,
                               num_warps=4)
    keys = keys.sort(-1).values
    U = NK if umax is None else min(NK, umax)
    ulist = torch.empty((rows, U), dtype=torch.int32, device=sel.device)
    umask = torch.zeros((rows, U), dtype=torch.int32, device=sel.device)
    ucount = torch.empty((rows,), dtype=torch.int32, device=sel.device)
    union_build_kernel[(rows,)](keys, ulist, umask, ucount, U, NK=NK, BQ=BQ, BIG=UNION_BIGV,
                                UP=triton.next_power_of_2(U), num_warps=8 if NK >= 4096 else 4)
    return ulist.view(B, NB, Hkv, U), umask.view(B, NB, Hkv, U), ucount.view(B, NB, Hkv)


def union_xmask_triton(next_sel, ulist, ucount, BQ, CR):
    B, T, Hkv, TK = next_sel.shape
    assert next_sel.stride(3) == 1 and next_sel.stride(2) == TK
    NB, U = ulist.shape[1], ulist.shape[-1]
    rows = B * NB * Hkv
    xm = torch.zeros((B, NB, Hkv, U * BQ), dtype=torch.int32, device=next_sel.device)
    union_xmask_kernel[(rows,)](next_sel, ulist, ucount, xm, T, NB, U, next_sel.stride(0), next_sel.stride(1),
                                H_kv=Hkv, TK=TK, BQ=BQ, CR=CR, NSTEP=max(1, int(U).bit_length() + 1), num_warps=4)
    return xm


def union_lists_fast(sel, BQ, umax=None):
    """union_lists through the CUDA builder (recsa/cuda/union_build.cu) when it applies (BQ 16, TK 256, sm_90)."""
    global _LEAF_CUDA_BROKEN
    if (LEAF_CUDA and not _LEAF_CUDA_BROKEN and BQ == 16 and sel.shape[-1] == 256 and sel.dtype == torch.int32
            and sel.stride(3) == 1 and sel.stride(2) == 256 and torch.cuda.get_device_capability(sel.device) == (9, 0)):
        from recsa import leaf_cuda as LC
        try:
            return LC.union_lists16(sel, pad=True, umax=umax, n_parents=umax)
        except Exception as ex:      # no nvcc / driver
            LC.cuda_fallback("union_build", ex)         # raises under strict mode
            _LEAF_CUDA_BROKEN = True
            print(f"[recsa] CUDA union builder unavailable, using the Triton one ({ex})", flush=True)
    return union_lists(sel, BQ, umax)


def union_lists(sel, BQ, umax=None):
    if sel.stride(3) == 1 and sel.stride(2) == sel.shape[3]:
        return union_lists_triton(sel, BQ, umax)
    return union_lists_torch(sel, BQ, umax)


def union_xmask(next_sel, ulist, BQ, CR, ucount=None, n_parents=None):
    if (n_parents is not None and ucount is not None and BQ == 16 and LEAF_CUDA and not _LEAF_CUDA_BROKEN
            and next_sel.dtype == torch.int32 and torch.cuda.get_device_capability(next_sel.device) == (9, 0)):
        from recsa import leaf_cuda as LC
        xm = LC.union_xmask16(next_sel, ulist, ucount, CR, n_parents)
        if xm is not None:
            return xm
    if ucount is not None and next_sel.stride(3) == 1 and next_sel.stride(2) == next_sel.shape[3]:
        return union_xmask_triton(next_sel, ulist, ucount, BQ, CR)
    return union_xmask_torch(next_sel, ulist, BQ, CR)
