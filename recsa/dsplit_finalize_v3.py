"""Decode select-finalize kernel (RECSA_DSPLIT_FINALIZE, the default)."""
import triton
import triton.language as tl


@triton.jit
def _fin3_bit_step(us, tau, nbit, TOP_K: tl.constexpr):
    """One greedy bisection step of the reference kernel on bit `nbit` (U domain), with the early-exit flag (count ==
    TOP_K)."""
    SIGN: tl.constexpr = -2147483648
    cand_t = tau | (1 << nbit)
    cnt = tl.sum((us >= (cand_t ^ SIGN)).to(tl.int32))
    tau = tl.where(cnt >= TOP_K, cand_t, tau)
    done = (cnt == TOP_K).to(tl.int32)
    return tau, nbit - 1, done


@triton.jit
def recsa_dsplit_select_finalize_v3_kernel(
    prev_selected_parents,  # [B, 1, H_kv, TOP_K]
    next_selected_parents,
    raw_imp,
    batch_m_in,  # [B*H_kv * CR] fp32 from the decode scoring kernel
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
    SEL_EPT: tl.constexpr = 0,
    SEL_ASM: tl.constexpr = "",
    SEL_CONS: tl.constexpr = "",
    SCAT_ASM: tl.constexpr = "",
):
    SIGN: tl.constexpr = -2147483648
    CR: tl.constexpr = COMPRESSION_RATE
    MAX_CAND: tl.constexpr = TOP_K * COMPRESSION_RATE
    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE
    tl.static_assert(BC * CR == MAX_CAND)

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
    out_par_base = prev_sel_base
    slot = tl.arange(0, MAX_CAND).to(tl.int32)
    slot64 = slot.to(tl.int64)
    par_all = tl.load(prev_selected_parents + prev_sel_base + (slot // COMPRESSION_RATE).to(tl.int64)).to(tl.int32)
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    rows = tl.arange(0, CR)
    cols = tl.arange(0, BC)
    s2 = (rows[:, None] * BC + cols[None, :]).to(tl.int32)
    v2 = tl.load(raw_imp + pid_bh_kv.to(tl.int64) * MAX_CAND + s2.to(tl.int64))
    bm_base = batch_m_in + pid_bh_kv.to(tl.int64) * COMPRESSION_RATE
    bm_raw = ()
    for i in tl.static_range(CR):
        bm_raw = bm_raw + (tl.load(bm_base + i),)
    if WRITE_TAIL:
        t_lanes = tl.arange(0, TOP_K).to(tl.int32)
        tl.store(next_selected_parents + out_par_base + t_lanes.to(tl.int64),
                 tl.full([TOP_K], -1, dtype=tl.int32))

    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))
    num_batches = (num_valid_parents + (PARENTS_PER_BATCH - 1)) // PARENTS_PER_BATCH
    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rightmost_pos = (num_candidates - 1).to(tl.int32)

    idx_mask = (1 << LOG_N) - 1

    running_M = tl.full([], NEG_INF, tl.float32)
    rb_vec = tl.zeros([CR], dtype=tl.float32)
    rps = ()
    for i in tl.static_range(CR):
        alive = i < num_batches
        bm_i = tl.where(alive, bm_raw[i], NEG_INF)
        new_M = tl.maximum(running_M, bm_i)
        rescale_prefix = tl.exp(running_M - new_M)
        rescale_batch = tl.exp(bm_i - new_M)
        rb_vec = tl.where(rows == i, rescale_batch, rb_vec)
        rps = rps + (rescale_prefix,)
        running_M = tl.where(alive, new_M, running_M)
    # v_new of every live row at once (row r with iteration r's factor)
    alive_row = rows < num_batches
    vn = tl.where(s2 < num_candidates, v2, NEG_INF)
    vn = tl.where(vn >= 0, vn * rb_vec[:, None], vn)
    vn = tl.where(s2 == rightmost_pos, 1e6, vn)
    vn = (vn.to(tl.int32, bitcast=True) & ~idx_mask).to(tl.float32, bitcast=True)
    v2 = tl.where(alive_row[:, None], vn, v2)
    # v_old passes in increasing i on rows < i; factor-1.0 passes are identities
    for i in tl.static_range(1, CR):
        rp = rps[i]
        if (i < num_batches) & (rp != 1.0):
            vo = tl.where(v2 >= 0, v2 * rp, v2)
            vo = (vo.to(tl.int32, bitcast=True) & ~idx_mask).to(tl.float32, bitcast=True)
            v2 = tl.where((rows < i)[:, None], vo, v2)
    v = tl.reshape(v2, [MAX_CAND])

    sbase = pid_bh_kv.to(tl.int64) * MAX_CAND
    if HAS_FORCE:
        f_node = tl.maximum(par_all, 0) * COMPRESSION_RATE + (slot % COMPRESSION_RATE)
        f_ok = (par_all >= 0) & (slot < num_candidates)
        f_word = tl.load(force_mask + (f_node >> 5).to(tl.int64), mask=f_ok, other=0)
        f_hit = ((f_word >> (f_node & 31)) & 1) != 0
        f_force = f_hit & f_ok
        fv = tl.where(f_force, 1e6, v)
        fb = fv.to(tl.int32, bitcast=True) & ~idx_mask
        v = tl.where(f_force, fb.to(tl.float32, bitcast=True), v)
    tl.store(scratch_imp + sbase + slot64, v)

    valid_c = slot < num_candidates
    vals = tl.where(valid_c, v, NEG_INF)
    vi = vals.to(tl.int32, bitcast=True)
    enc_idx = tl.where(vals >= 0, ~slot, slot) & idx_mask
    enc = (vi & ~idx_mask) | enc_idx
    us = (enc ^ ((enc >> 31) | SIGN)) ^ SIGN
    if SEL_EPT > 0:
        tau_addr = (scratch_tau + pid_bh_kv.to(tl.int64)).to(tl.int64)
        taus = tl.inline_asm_elementwise(SEL_ASM, SEL_CONS, [us, slot, tau_addr], dtype=tl.int32, is_pure=False,
                                         pack=SEL_EPT)
    else:
        tau = tl.full([], 0, tl.int32)  # decided prefix, U domain
        nbit = tl.full([], 31, tl.int32)  # next undecided bit
        done = tl.full([], 0, tl.int32)
        while (done == 0) & (nbit >= 0):
            tau, nbit, done = _fin3_bit_step(us, tau, nbit, TOP_K)
        taus = tau ^ SIGN
        if done != 0:
            # exactly TOP_K keys are >= tau: the greedy result is their minimum
            taus = tl.min(tl.where(us >= taus, us, 2147483647))
        tl.store(scratch_tau + pid_bh_kv.to(tl.int64), taus)

    # ---- emit selected; rank = kept slots before it (slot order) ----
    keep = (us >= taus) & (vals > SCORE_VALID_THRESHOLD) & valid_c
    child_slot_f = slot % COMPRESSION_RATE
    parent_idx_f = tl.where(keep, par_all, -1)
    node = tl.where(keep & (parent_idx_f >= 0),
                    parent_idx_f * COMPRESSION_RATE + child_slot_f, -1)
    ok = node >= 0
    oki = ok.to(tl.int32)
    rk = tl.cumsum(oki, axis=0) - oki
    if WRITE_TAIL:
        tl.debug_barrier()
    if SEL_EPT > 0:
        out_addr = (next_selected_parents + out_par_base).to(tl.int64)
        tl.inline_asm_elementwise(SCAT_ASM, "=r,r,r,l", [node, rk, out_addr], dtype=tl.int32, is_pure=False, pack=1)
    else:
        tl.store(next_selected_parents + out_par_base + rk.to(tl.int64), node,
                 mask=ok & (rk < TOP_K))


def sel_ptx(ept: int, top_k: int, max_cand: int) -> str:
    """The kernel's greedy 1-bit bisection as inline PTX, ept = MAX_CAND / threads keys per thread. Each step counts
    keys >= thresh with redux.sync, then sums the warp partials in double-buffered shared memory after one bar.sync."""
    nw = max_cand // (32 * ept)
    if nw * 32 * ept != max_cand or nw % 4 or not 4 <= nw <= 32 or ept & (ept - 1):
        raise ValueError(f"sel_ptx: unsupported ept={ept} max_cand={max_cand}")
    K = [f"${ept + i}" for i in range(ept)]
    S = [f"${2 * ept + i}" for i in range(ept)]
    TAU_PTR = f"${3 * ept}"
    s1 = (max_cand * (max_cand - 1) // 2) & 0xFFFFFFFF
    s2 = ((max_cand - 1) * max_cand * (2 * max_cand - 1) // 6) & 0xFFFFFFFF
    CK = 256
    L = []
    a = L.append
    a("{")
    a(".reg .pred fq_p, fq_l0, fq_t0p;")
    a(".reg .b32 fq_tid, fq_w, fq_x, fq_base, fq_sta, fq_lda, fq_tau, fq_bit, fq_cand, fq_th, fq_c, "
      "fq_part, fq_tot, fq_t0, fq_t1, fq_t2, fq_t3, fq_res, fq_s1, fq_s2, fq_q1, fq_q2;")
    a(".reg .b32 " + ", ".join(f"fq_k{i}" for i in range(ept)) + ";")
    a(f".shared .align 16 .b32 fq_red[{(CK + 16 * nw) // 4}];")
    a("mov.u32 fq_tid, %tid.x;")
    a("mov.u32 fq_x, %ntid.x;")
    a(f"mul.lo.u32 fq_x, fq_x, {ept};")
    a(f"setp.ne.u32 fq_p, fq_x, {max_cand};")
    a("@fq_p trap;")
    a("shr.u32 fq_w, fq_tid, 5;")
    a("and.b32 fq_x, fq_tid, 31;")
    a("setp.eq.u32 fq_l0, fq_x, 0;")
    a("setp.eq.u32 fq_t0p, fq_tid, 0;")
    a("mov.u32 fq_base, fq_red;")
    a("shl.b32 fq_sta, fq_w, 2;")
    a("add.u32 fq_sta, fq_sta, fq_base;")  # this warp's slot in buffer 0; buffer 1 = +128
    a("mov.b32 fq_lda, fq_base;")

    def count_ge(th):
        for i, k in enumerate(K):
            a(f"set.ge.u32.s32 fq_k{i}, {k}, {th};")
        n = ept
        while n > 1:
            for i in range(n // 2):
                a(f"add.u32 fq_k{i}, fq_k{2 * i}, fq_k{2 * i + 1};")
            n //= 2
        a("neg.s32 fq_c, fq_k0;")

    def exchange(red):
        a("@fq_l0 st.shared.u32 [fq_sta], fq_part;")
        a("bar.sync 0;")
        for j in range(nw // 4):
            a(f"ld.shared.v4.u32 {{fq_t0, fq_t1, fq_t2, fq_t3}}, [fq_lda+{16 * j}];")
            a(f"{red} fq_t0, fq_t0, fq_t1;")
            a(f"{red} fq_t2, fq_t2, fq_t3;")
            a(f"{red} fq_t0, fq_t0, fq_t2;")
            a(f"{red} fq_tot, fq_tot, fq_t0;" if j else "mov.b32 fq_tot, fq_t0;")
        a("xor.b32 fq_sta, fq_sta, 128;")
        a("xor.b32 fq_lda, fq_lda, 128;")

    a(f"mov.b32 fq_s1, {S[0]};")
    a(f"mul.lo.u32 fq_s2, {S[0]}, {S[0]};")
    for sl in S[1:]:
        a(f"add.u32 fq_s1, fq_s1, {sl};")
        a(f"mad.lo.u32 fq_s2, {sl}, {sl}, fq_s2;")
    a("redux.sync.add.u32 fq_s1, fq_s1, 0xffffffff;")
    a("redux.sync.add.u32 fq_s2, fq_s2, 0xffffffff;")
    a("mov.b32 fq_th, 0;")
    count_ge("fq_th")
    a("redux.sync.add.u32 fq_part, fq_c, 0xffffffff;")
    a("shl.b32 fq_x, fq_w, 4;")
    a("add.u32 fq_x, fq_x, fq_base;")
    a(f"@fq_l0 st.shared.v4.u32 [fq_x+{CK}], {{fq_part, fq_s1, fq_s2, fq_s2}};")
    a("bar.sync 0;")
    for j in range(nw):
        a(f"ld.shared.v4.u32 {{fq_t0, fq_t1, fq_t2, fq_t3}}, [fq_base+{CK + 16 * j}];")
        if j == 0:
            a("mov.b32 fq_tot, fq_t0;")
            a("mov.b32 fq_q1, fq_t1;")
            a("mov.b32 fq_q2, fq_t2;")
        else:
            a("add.u32 fq_tot, fq_tot, fq_t0;")
            a("add.u32 fq_q1, fq_q1, fq_t1;")
            a("add.u32 fq_q2, fq_q2, fq_t2;")
    a(f"setp.ne.u32 fq_p, fq_q1, {s1};")
    a("@fq_p trap;")
    a(f"setp.ne.u32 fq_p, fq_q2, {s2};")
    a("@fq_p trap;")
    a(f"setp.ge.s32 fq_p, fq_tot, {top_k};")
    a("selp.b32 fq_tau, 0x80000000, 0, fq_p;")
    a(f"setp.eq.s32 fq_p, fq_tot, {top_k};")
    a("@fq_p bra.uni FQEARLY${:uid};")
    a("mov.b32 fq_bit, 0x40000000;")
    # ---- bits 30 .. 0 ----
    a("FQLOOP${:uid}:")
    a("or.b32 fq_cand, fq_tau, fq_bit;")
    a("xor.b32 fq_th, fq_cand, 0x80000000;")
    count_ge("fq_th")
    a("redux.sync.add.u32 fq_part, fq_c, 0xffffffff;")
    exchange("add.u32")
    a(f"setp.ge.s32 fq_p, fq_tot, {top_k};")
    a("selp.b32 fq_tau, fq_cand, fq_tau, fq_p;")
    a(f"setp.eq.s32 fq_p, fq_tot, {top_k};")
    a("@fq_p bra.uni FQEARLY${:uid};")
    a("shr.u32 fq_bit, fq_bit, 1;")
    a("setp.ne.u32 fq_p, fq_bit, 0;")
    a("@fq_p bra.uni FQLOOP${:uid};")
    a("xor.b32 fq_res, fq_tau, 0x80000000;")
    a("bra.uni FQDONE${:uid};")
    a("FQEARLY${:uid}:")
    a("xor.b32 fq_th, fq_tau, 0x80000000;")
    a("mov.b32 fq_c, 0x7fffffff;")
    for k in K:
        a(f"setp.ge.s32 fq_p, {k}, fq_th;")
        a(f"@fq_p min.s32 fq_c, fq_c, {k};")
    a("redux.sync.min.s32 fq_part, fq_c, 0xffffffff;")
    exchange("min.s32")
    a("mov.b32 fq_res, fq_tot;")
    a("FQDONE${:uid}:")
    a(f"@fq_t0p st.global.b32 [{TAU_PTR}], fq_res;")
    for i in range(ept):
        a(f"mov.b32 ${i}, fq_res;")
    a("}")
    return "\n".join(L)


def scat_ptx(top_k: int) -> str:
    """Predicated scatter: next[rank] = node if node >= 0 and rank < TOP_K."""
    return ("{\n.reg .pred fs_p;\n.reg .b64 fs_a;\n"
            "setp.ge.s32 fs_p, $1, 0;\n"
            f"setp.lt.and.s32 fs_p, $2, {top_k}, fs_p;\n"
            "mul.wide.s32 fs_a, $2, 4;\n"
            "add.s64 fs_a, fs_a, $3;\n"
            "@fs_p st.global.b32 [fs_a], $1;\n"
            "mov.b32 $0, 0;\n}")


_SEL_CACHE = {}


def sel_kwargs(top_k: int, compression_rate: int, num_warps: int, ptx: bool = True) -> dict:
    """Selection / scatter constexprs: the PTX path when ptx is set and the geometry fits (ept a power of two,
    4 <= num_warps <= 32, num_warps % 4 == 0), else the Triton path (SEL_EPT=0)."""
    max_cand = top_k * compression_rate
    ept = max_cand // (32 * num_warps)
    if (not ptx or ept < 1 or ept & (ept - 1) or ept * 32 * num_warps != max_cand or num_warps % 4
            or not 4 <= num_warps <= 32):
        return dict(SEL_EPT=0)
    key = (ept, top_k, max_cand)
    if key not in _SEL_CACHE:
        _SEL_CACHE[key] = dict(SEL_EPT=ept, SEL_ASM=sel_ptx(ept, top_k, max_cand),
                               SEL_CONS=",".join(["=r"] * ept + ["r"] * (2 * ept) + ["l"] * ept),
                               SCAT_ASM=scat_ptx(top_k))
    return _SEL_CACHE[key]
