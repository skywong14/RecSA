"""Second fused MLP-pooler K pool for decode: the arithmetic of the fused kernel in recsa/mlp_pool_exact.py (the base
kernel) with different data movement."""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from recsa.mlp_pool_exact import (FDBG_SIZE_C, FLT_LOWEST_C, _LAUNCH, _halve, _jtile, _mv_emul, _r1_tile, _r4_tile,
                                _seq16, _warp_tree, _z1_emul, _z2_emul, clamp_min_nan, div_rn, expf_nf, fma_rn,
                                silu_f32, sqrt_rn)

RECSA_MLP_POOL_FUSED2 = os.environ.get("RECSA_MLP_POOL_FUSED2", "on").strip().lower()
if RECSA_MLP_POOL_FUSED2 not in ("on", "off"):
    raise ValueError(f"RECSA_MLP_POOL_FUSED2 must be 'on' or 'off', got {RECSA_MLP_POOL_FUSED2!r}")
V2_WARPS = 2


# ---------------------------------------------------------------- primitives

@triton.jit
def _gt(dep):
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.s32 p, $1, -1234567; mov.u64 $0, 0; @p mov.u64 $0, %globaltimer; }",
        "=l,r", [dep.to(tl.int32)], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _warm_sectors(ptr, N: tl.constexpr):
    return tl.load(ptr + tl.arange(0, N) * 8)


@triton.jit
def _sink(x, sink_ptr, sink_on):
    # keeps warm loads alive: a store that never executes (sink_on == 0 at runtime)
    tl.store(sink_ptr + tl.zeros(x.shape, tl.int32), x, mask=sink_on != 0)


@triton.jit
def _shfl_down(x, delta):
    # lane L receives lane L + delta's x (lanes past 31 keep their own: clamp 0x1f)
    return tl.inline_asm_elementwise("shfl.sync.down.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=r,r,r", [x, delta],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _shfl_idx(x, src):
    # every lane receives lane src's x
    return tl.inline_asm_elementwise("shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=r,r,r", [x, src],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _jidx():
    q = tl.arange(0, 16)
    return 8 * ((q // 2) % 2) + 4 * (q % 2) + 2 * ((q // 4) % 2) + q // 8


@triton.jit
def _xwarp(x, d, jj, scr_ptr):
    tl.store(scr_ptr + jj * 64 + d, x)
    tl.debug_barrier()
    y = tl.load(scr_ptr + jj * 64 + (d + 32) % 64)
    lo = tl.where(d < 32, x, y)
    hi = tl.where(d < 32, y, x)
    return lo, hi


@triton.jit
def _tree64(t, d, scr_ptr):
    lo, hi = _xwarp(t, d, tl.zeros([1, 1], tl.int32), scr_ptr)
    u = lo + hi
    u = u + _shfl_down(u, 1)
    u = u + _shfl_down(u, 2)
    u = u + _shfl_down(u, 4)
    u = u + _shfl_down(u, 8)
    u = u + _shfl_down(u, 16)
    return _shfl_idx(u, 0)


@triton.jit
def _mv_shfl(dv0, dv1, d, pw_ptr, scr_ptr):
    g = (d // 4) * 4
    a = fma_rn(dv0, tl.load(pw_ptr + g), tl.zeros(dv0.shape, tl.float32))
    a = fma_rn(_shfl_down(dv0, 1), tl.load(pw_ptr + g + 1), a)
    a = fma_rn(_shfl_down(dv0, 2), tl.load(pw_ptr + g + 2), a)
    a = fma_rn(_shfl_down(dv0, 3), tl.load(pw_ptr + g + 3), a)
    b = fma_rn(dv1, tl.load(pw_ptr + 64 + g), tl.zeros(dv1.shape, tl.float32))
    b = fma_rn(_shfl_down(dv1, 1), tl.load(pw_ptr + 64 + g + 1), b)
    b = fma_rn(_shfl_down(dv1, 2), tl.load(pw_ptr + 64 + g + 2), b)
    b = fma_rn(_shfl_down(dv1, 3), tl.load(pw_ptr + 64 + g + 3), b)
    c = tl.reshape(a + b, (64, 16))
    lo, hi = _xwarp(c, tl.reshape(d, (64, 1)), tl.reshape(tl.arange(0, 16), (1, 16)), scr_ptr)
    c = lo + hi                                          # offset 8
    c = c + _shfl_down(c, 16)                            # offset 4
    c = c + _shfl_down(c, 8)                             # offset 2
    c = c + _shfl_down(c, 4)                             # offset 1
    return c + 0.0


@triton.jit
def recsa_mlp_pool_v2_trees_kernel(
    x_ptr,
    idx_ptr,   # int64 [16]: 0..15
    pw_ptr,    # fp32 [128]
    t_ptr,     # fp32 [H, 64]: per-lane var / mean operands t_x
    xs_ptr,
    dev_ptr,
    mv1_ptr,   # fp32 [16, H] OUT: _mv_emul (base kernel)
    mv2_ptr,   # fp32 [16, H] OUT: _mv_shfl (this module)
    r1_ptr,    # fp32 [H] OUT: _warp_tree(_halve(t)) (base kernel)
    r2_ptr,    # fp32 [H] OUT: _tree64(t) (this module)
    H: tl.constexpr,
):
    """Test kernel: this module's shuffle trees and the base kernel's helpers on the same operands."""
    h = tl.program_id(0)
    ROW: tl.constexpr = H * 128
    j = _jtile()
    d = tl.reshape(tl.arange(0, 64), (64, 1, 1, 1, 1))
    o0 = h * 128 + d
    rows = tl.load(idx_ptr + j)
    x0 = tl.load(x_ptr + rows * ROW + o0)
    x1 = tl.load(x_ptr + rows * ROW + o0 + 64)
    d64 = tl.reshape(tl.arange(0, 64), (64, 1))
    mvc = _mv_shfl(x0, x1, d, pw_ptr, xs_ptr + h * 2048)
    tl.store(mv2_ptr + tl.reshape(_jidx(), (1, 16)) * H + h + 0 * d64, mvc, mask=d64 == 0)
    tl.store(dev_ptr + j * ROW + o0, x0)
    tl.store(dev_ptr + j * ROW + o0 + 64, x1)
    tl.debug_barrier()
    _mv_emul(dev_ptr, pw_ptr, mv1_ptr, h, H)
    t = tl.load(t_ptr + h * 64 + tl.arange(0, 64))
    r2 = _tree64(tl.reshape(t, (64, 1)), d64, xs_ptr + h * 2048 + 1024)
    tl.store(r2_ptr + h + 0 * d64, r2, mask=d64 == 0)
    r1 = _warp_tree(_halve(tl.reshape(t, (1, 64)), 1, 64), 1)
    tl.store(r1_ptr + h + tl.zeros([1, 1], tl.int32), r1)


def trees_v1_v2(x: torch.Tensor, pw: torch.Tensor, t: torch.Tensor):
    """Runs recsa_mlp_pool_v2_trees_kernel on x [16, 8, 128], pw [128] and t [8, 64] (fp32); returns
    (mv_v1, mv_v2, r_v1, r_v2)."""
    dev = x.device
    Hh = x.shape[1]
    idx = torch.arange(16, device=dev)
    xs = torch.empty(Hh * 2048, device=dev)
    dv = torch.empty(16 * Hh * 128, device=dev)
    mv1, mv2 = torch.empty(16 * Hh, device=dev), torch.empty(16 * Hh, device=dev)
    r1, r2 = torch.empty(Hh, device=dev), torch.empty(Hh, device=dev)
    with torch.cuda.device(dev):
        recsa_mlp_pool_v2_trees_kernel[(Hh,)](x, idx, pw, t, xs, dv, mv1, mv2, r1, r2, H=Hh, num_warps=V2_WARPS,
                                              **_LAUNCH)
    return mv1, mv2, r1, r2


# ---------------------------------------------------------------- the kernel

@triton.jit
def _v2_level(ck, ip, vp, cp, ep, pw_ptr, w1_ptr, b1_ptr, w2_ptr, tau, gate, xs_ptr, sig_ptr, h_ptr, hid_ptr,
              kp0_ptr, mv_ptr, sm_ptr, ok_ptr, pp, leaf_v, k_t, v_t, t_ptr, is_leaf, warm2, ip2, sink_ptr, sink_on,
              dbg_ptr, ts_ptr, h, H: tl.constexpr, H1: tl.constexpr, LEAF: tl.constexpr, DBG: tl.constexpr,
              TS: tl.constexpr):
    ROW: tl.constexpr = H * 128
    if TS:
        tl.store(ts_ptr + 0, _gt(h))
    j = _jtile()                                         # [1, 2, 2, 2, 2]
    d = tl.reshape(tl.arange(0, 64), (64, 1, 1, 1, 1))
    o0 = h * 128 + d
    o1 = o0 + 64
    rows = tl.load(ip + j)
    ok = tl.load(vp + j) == 0
    x0 = tl.load(ck + rows * ROW + o0)
    x1 = tl.load(ck + rows * ROW + o1)
    r2 = tl.load(ip2 + j, mask=warm2, other=0)
    SEC: tl.constexpr = 128 * ok_ptr.dtype.element_ty.primitive_bitwidth // 256      # 32-byte sectors per head row
    sec = tl.reshape(tl.arange(0, SEC), (1, 1, 1, 1, 1, SEC))
    rw = tl.load(ok_ptr + tl.reshape(r2, (1, 2, 2, 2, 2, 1)) * ROW + h * 128
                 + sec * (256 // ok_ptr.dtype.element_ty.primitive_bitwidth), mask=warm2, other=0.0)
    if LEAF:
        t_row = tl.load(t_ptr)
        kt0 = tl.load(k_t + o0)
        kt1 = tl.load(k_t + o1)
        sub = (rows == t_row) & is_leaf
        x0 = tl.where(sub, kt0, x0)
        x1 = tl.where(sub, kt1, x1)
        ol = h * 128 + tl.reshape(tl.arange(0, 64), (1, 64))
        tl.store(ck + t_row * ROW + ol, tl.reshape(kt0, (1, 64)), mask=is_leaf)
        tl.store(ck + t_row * ROW + ol + 64, tl.reshape(kt1, (1, 64)), mask=is_leaf)
        tl.store(leaf_v + t_row * ROW + ol, tl.load(v_t + ol), mask=is_leaf)
        tl.store(leaf_v + t_row * ROW + ol + 64, tl.load(v_t + ol + 64), mask=is_leaf)
    # ---------------------------------------------------------- seg A
    x0 = tl.where(ok, x0.to(tl.float32), 0.0)
    x1 = tl.where(ok, x1.to(tl.float32), 0.0)
    cnt = tl.load(cp)
    den = tl.load(ep)
    mu0 = div_rn(_r1_tile(x0), cnt)                      # [64]
    mu1 = div_rn(_r1_tile(x1), cnt)
    dv0 = tl.where(ok, x0 - tl.reshape(mu0, (64, 1, 1, 1, 1)), 0.0)
    dv1 = tl.where(ok, x1 - tl.reshape(mu1, (64, 1, 1, 1, 1)), 0.0)
    ev0, od0 = _r4_tile(dv0 * dv0)
    ev1, od1 = _r4_tile(dv1 * dv1)
    t = ((ev0 + ev1) + od0) + od1
    d64 = tl.reshape(tl.arange(0, 64), (64, 1))
    vs = _tree64(tl.reshape(t, (64, 1)), d64, xs_ptr + 1024)             # [64, 1], all equal
    tl.store(sig_ptr + h + 0 * d64, sqrt_rn(clamp_min_nan(div_rn(vs, den), 1e-12)), mask=d64 == 0)
    _sink(rw.to(tl.float32), sink_ptr, sink_on)
    if TS:
        tl.store(ts_ptr + 1, _gt(h))
    # ---------------------------------------------------------- mv (cuBLAS order)
    mvc = _mv_shfl(dv0, dv1, d, pw_ptr, xs_ptr)                          # [64, 16 (j')], valid at d == 0
    tl.store(mv_ptr + tl.reshape(_jidx(), (1, 16)) * H + h + 0 * d64, mvc, mask=d64 == 0)
    tl.debug_barrier()
    if TS:
        tl.store(ts_ptr + 2, _gt(h))
    if DBG:
        tl.store(dbg_ptr + j * H + h, tl.load(mv_ptr + j * H + h))
    sig = tl.load(sig_ptr + h)
    lane = tl.arange(0, 64)
    jv = tl.arange(0, 16)
    okv = tl.load(vp + jv) == 0
    mvv = tl.load(mv_ptr + jv * H + h)
    lmv = tl.where(okv, div_rn(tau * mvv, sig), float("-inf"))
    mx = tl.maximum(tl.max(tl.where(lmv != lmv, FLT_LOWEST_C, lmv), axis=0), FLT_LOWEST_C)
    exv = expf_nf(lmv - mx)
    tl.store(sm_ptr + h * 32 + jv, exv)
    tl.debug_barrier()
    if TS:
        tl.store(ts_ptr + 3, _gt(h))
    s = _seq16(tl.load(sm_ptr + h * 32 + j))             # [1]
    tl.store(sm_ptr + h * 32 + 16 + jv, div_rn(exv, s))
    tl.debug_barrier()
    if TS:
        tl.store(ts_ptr + 4, _gt(h))
    alpha = tl.load(sm_ptr + h * 32 + 16 + j)
    p0 = _r1_tile(alpha * x0)                            # [64]
    p1 = _r1_tile(alpha * x1)
    d0 = p0 - mu0
    d1 = p1 - mu1
    tl.store(kp0_ptr + h * 128 + lane, mu0 + gate * d0)
    tl.store(kp0_ptr + h * 128 + 64 + lane, mu1 + gate * d1)
    q0 = tl.reshape(mu0 * mu0, (64, 1))
    q1 = tl.reshape(mu1 * mu1, (64, 1))
    z = tl.zeros([64, 1], tl.float32)
    tq = (((z + q0) + (z + q1)) + 0.0) + 0.0
    rmean = _tree64(tq, d64, xs_ptr + 1088) * 0.0078125
    rho = tl.reshape(sqrt_rn(clamp_min_nan(rmean, 1e-12)), (64,))
    hb = h_ptr + h * 256 + lane
    tl.store(hb, div_rn(mu0, rho))
    tl.store(hb + 64, div_rn(mu1, rho))
    tl.store(hb + 128, div_rn(d0, sig))
    tl.store(hb + 192, div_rn(d1, sig))
    tl.debug_barrier()
    if TS:
        tl.store(ts_ptr + 5, _gt(h))
    n1 = tl.arange(0, H1)
    z1 = _z1_emul(h_ptr + h * 256, w1_ptr, H1)           # [H1]
    if DBG:
        tl.store(dbg_ptr + 16 * H + h * H1 + n1, z1)
    zb = z1 + tl.load(b1_ptr + n1)
    tl.store(hid_ptr + h * H1 + n1, silu_f32(zb))
    tl.debug_barrier()
    if TS:
        tl.store(ts_ptr + 6, _gt(h))
    n2 = tl.arange(0, 128)
    z2 = _z2_emul(hid_ptr + h * H1, w2_ptr, H1)          # [128]
    if DBG:
        tl.store(dbg_ptr + 16 * H + H * H1 + h * 128 + n2, z2)
    k = tl.load(kp0_ptr + h * 128 + n2) + sig * z2
    p = tl.load(pp)
    tl.store(ok_ptr + p * (H * 128) + h * 128 + n2, k.to(ok_ptr.dtype.element_ty))
    if TS:
        tl.store(ts_ptr + 15, _gt(p.to(tl.int32)))


@triton.jit(do_not_specialize=["n_lev", "dk", "di", "dv", "dc", "de", "do", "dp"])
def recsa_mlp_pool_fused_v2_kernel(
    child_k,
    idx_ptr,   # int64 [16]: level l's window child rows
    inv_ptr,   # uint8 [16]: its pad mask (j >= n)
    cnt_ptr,   # fp32 [1]: float(n)
    den_ptr,   # fp32 [1]: float(n * 128)
    pw_ptr,    # fp32 [128]: probe_w
    w1_ptr,    # fp32 [256, H1]: mlp_w1^T (fused_weights)
    b1_ptr,    # fp32 [H1]: mlp_b1
    w2_ptr,    # fp32 [H1, 128]: mlp_w2^T (fused_weights)
    tau,       # fp32 host scalar
    gate,      # fp32 host scalar
    xs_ptr,
    sig_ptr,   # fp32 [H] scratch
    h_ptr,     # fp32 [H, 256] scratch
    hid_ptr,   # fp32 [H, H1] scratch
    kp0_ptr,   # fp32 [H, 128] scratch
    mv_ptr,    # fp32 [16, H] scratch
    sm_ptr,    # fp32 [H, 32] scratch
    out_k,
    p_ptr,     # int64 [1]: level l's open parent slot
    leaf_v,    # [1, cap, H, 128] level-0 V buffer (LEAF only)
    k_t,       # [H, 128] the new leaf K row (LEAF only)
    v_t,       # [H, 128] the new leaf V row (LEAF only)
    t_ptr,
    n_lev,     # 1 or 2: levels l (.. l + 1) pooled by this launch
    dk, di, dv, dc, de, do, dp,
    dbg_ptr,
    sink_ptr,  # fp32 [1]: the warm loads' never-executed store
    sink_on,   # int: 0
    ts_ptr,
    H: tl.constexpr, H1: tl.constexpr, LEAF: tl.constexpr, DBG: tl.constexpr = False, TS: tl.constexpr = False,
):
    """grid (H,). n_lev = 1: level l; n_lev = 2: also level l + 1 (child buffer out_k) in the same program."""
    h = tl.program_id(0)
    xs = xs_ptr + h * 2048
    ws1 = _warm_sectors(w1_ptr, 256 * H1 // 8)
    ws2 = _warm_sectors(w2_ptr, H1 * 128 // 8)
    ws3 = _warm_sectors(pw_ptr, 16)
    ws4 = _warm_sectors(b1_ptr, H1 // 8)
    for lvl in range(0, n_lev):
        _v2_level(child_k + lvl * dk, idx_ptr + lvl * di, inv_ptr + lvl * dv, cnt_ptr + lvl * dc,
                  den_ptr + lvl * de, pw_ptr, w1_ptr, b1_ptr, w2_ptr, tau, gate, xs, sig_ptr, h_ptr, hid_ptr,
                  kp0_ptr, mv_ptr, sm_ptr, out_k + lvl * do, p_ptr + lvl * dp, leaf_v, k_t, v_t, t_ptr, lvl == 0,
                  (lvl == 0) & (n_lev > 1), idx_ptr + di, sink_ptr, sink_on, dbg_ptr + lvl * (FDBG_SIZE_C * H),
                  ts_ptr + lvl * (8 * 16) + h * 16, h, H, H1, LEAF, DBG, TS)
        tl.debug_barrier()
    _sink(ws1, sink_ptr, sink_on)
    _sink(ws2, sink_ptr, sink_on)
    _sink(ws3, sink_ptr, sink_on)
    _sink(ws4, sink_ptr, sink_on)


# ---------------------------------------------------------------- host side

_SINK: dict = {}


def _sink_buf(device):
    b = _SINK.get(device)
    if b is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("mlp_pool_fused_v2: the sink buffer must be allocated before CUDA-graph capture")
        b = _SINK[device] = torch.zeros(1, dtype=torch.float32, device=device)
    return b


def _delta(a: torch.Tensor, b: torch.Tensor) -> int:
    """b - a in elements (two operands of the same dtype)."""
    if a.dtype != b.dtype:
        raise ValueError("pool_k_fused_v2: per-level operands of different dtypes")
    nb = b.data_ptr() - a.data_ptr()
    if nb % a.element_size():
        raise ValueError("pool_k_fused_v2: per-level operands not element-aligned")
    return nb // a.element_size()


def pool_k_fused_v2(scr, child_k, win, probe_w, tau, gate, probe_extra, out_k, leaf=None, win2=None, out2_k=None,
                    wt=None, ts=None):
    """Drop-in for recsa.mlp_pool_exact.pool_k_fused using this module's kernel. ts, if given, receives the TS taps
    (int64 [2 * 8 * 16])."""
    from recsa.mlp_pool_exact import fused_weights
    h_kv = child_k.shape[2]
    if h_kv != 8 or child_k.shape[3] != 128:
        raise ValueError("pool_k_fused_v2: the deployed bucket shape only (H_kv = 8, head dim 128)")
    b1 = probe_extra["mlp_b1"]
    w1, w2 = fused_weights(probe_extra) if wt is None else wt
    inv = win.invalid4.view(torch.uint8)
    dbg = scr.fdbg if scr.fdbg is not None else scr.sig
    on = scr.fdbg is not None
    two = win2 is not None
    if two != (out2_k is not None):
        raise ValueError("pool_k_fused_v2: win2 and out2_k go together")
    if leaf is None:
        leaf_v, k_t, v_t, t_dev = child_k, child_k, child_k, win.p
    else:
        leaf_v, k_t, v_t, t_dev = leaf
    if two:
        dl = (_delta(child_k, out_k), _delta(win.idx, win2.idx), _delta(inv, win2.invalid4.view(torch.uint8)),
              _delta(win.cnt4, win2.cnt4), _delta(win.den3, win2.den3), _delta(out_k, out2_k), _delta(win.p, win2.p))
    else:
        dl = (0,) * 7
    sink = _sink_buf(child_k.device)
    with torch.cuda.device(child_k.device):
        return recsa_mlp_pool_fused_v2_kernel[(h_kv,)](
            child_k, win.idx, inv, win.cnt4, win.den3, probe_w, w1, b1, w2, tau, gate,
            scr.dev, scr.sig, scr.h, scr.hid, scr.kp0, scr.mv, scr.sm, out_k, win.p,
            leaf_v, k_t, v_t, t_dev, 2 if two else 1, *dl, dbg, sink, 0, sink if ts is None else ts,
            H=h_kv, H1=w1.shape[1], LEAF=leaf is not None, DBG=on, TS=ts is not None,
            num_warps=V2_WARPS, **_LAUNCH)


_CANARY: dict = {}
_warned = False
_CANARY_T = (0, 5, 15, 16 * 7 + 3, 16 * 15 + 15, 16 * 16 * 4 + 8)


def _canary(device, dtype, leaf: bool, two: bool) -> bool:
    """Compare this kernel with the base kernel on random operands for one specialisation (tree dtype, LEAF, two
    levels)."""
    from recsa import mlp_pool_exact as X
    from recsa.probe_pool import BucketCursor
    C, H, D, H1 = X.C_EXACT, X.H_EXACT, X.D_EXACT, 64
    g = torch.Generator(device=device)
    g.manual_seed(20261025)
    cap = C * C * 4 + 9
    rows = [(cap + C ** lv - 1) // C ** lv + 1 for lv in range(3)]
    t_dev = torch.zeros(1, dtype=torch.int64, device=device)
    win = BucketCursor(t_dev, C, D, cap, fused=True)
    scr = X.ExactScratch(H, D, C, H1, device)
    pw = torch.randn(D, generator=g, device=device)
    ex = {"mlp_w1": torch.randn(H1, 2 * D, generator=g, device=device) * 0.1,
          "mlp_b1": torch.randn(H1, generator=g, device=device),
          "mlp_w2": torch.randn(D, H1, generator=g, device=device) * 0.1}
    wt = X.fused_weights(ex)
    ok = True
    for fam in range(4):
        x = torch.randn(16, H, D, generator=g, device=device)
        pwc = torch.randn(D, generator=g, device=device)
        tt = torch.randn(H, 64, generator=g, device=device)
        if fam == 1:
            x = -x.abs() * 0.0
        elif fam == 2:
            x, pwc, tt = -x.abs() * 1e-25, pwc.abs() * 1e-25, -tt.abs() * 1e-45
        elif fam == 3:
            b = torch.randint(-(1 << 31), 1 << 31, (16 * H * D + H * 64,), generator=g, device=device,
                              dtype=torch.int64).to(torch.int32).view(torch.float32)
            b = torch.where(torch.isfinite(b) & (b.abs() < 1e18), b, 1.0)
            x, tt = b[:16 * H * D].view(16, H, D), b[16 * H * D:].view(H, 64)
        m1, m2, r1, r2 = trees_v1_v2(x.contiguous(), pwc.contiguous(), tt.contiguous())
        ok = ok and torch.equal(m1.view(torch.int32), m2.view(torch.int32)) and torch.equal(r1.view(torch.int32),
                                                                                          r2.view(torch.int32))
    for i, t in enumerate(_CANARY_T):
        t_dev.fill_(t)
        win.refresh(3)
        ks = []
        for r in rows:
            x = torch.randn(1, r, H, D, generator=g, device=device) * 3
            if i % 2:
                m = torch.rand(x.shape, generator=g, device=device)
                x = torch.where(m < 0.3, torch.where(m < 0.15, 0.0, -0.0), torch.where(m < 0.6, x * 1e-30, x.round()))
            x[:, -1].zero_()
            ks.append(x.to(dtype))
        v0 = torch.randn(1, rows[0], H, D, generator=g, device=device).to(dtype)
        kt = torch.randn(1, 1, H, D, generator=g, device=device).to(dtype)
        vt = torch.randn(1, 1, H, D, generator=g, device=device).to(dtype)
        outs = []
        for use_v2 in (False, True):
            k = [b.clone() for b in ks]
            v = v0.clone()
            lf = (v, kt[0, 0], vt[0, 0], t_dev) if leaf else None
            args = (scr, k[0], win[1], pw, 0.75, 0.375, ex, k[1], lf, win[2] if two else None,
                    k[2] if two else None, wt)
            (pool_k_fused_v2 if use_v2 else X._pool_k_fused_v1)(*args)
            outs.append(k + [v])
        torch.cuda.synchronize(device)
        for a, b in zip(*outs):
            iv = torch.int16 if a.element_size() == 2 else torch.int32
            ok = ok and torch.equal(a.view(iv), b.view(iv))
    return ok


def v2_ok(child_k, out_k, leaf, win2, out2_k) -> bool:
    """Whether this kernel can take the fused call: default bucket shape, one tree dtype for all levels, and a passing
    bind-time check (cached per device / dtype / LEAF / two levels)."""
    global _warned
    if child_k.shape[2] != 8 or child_k.shape[3] != 128 or out_k.dtype != child_k.dtype or (
            out2_k is not None and out2_k.dtype != child_k.dtype):
        return False
    dev = child_k.device
    key = (dev.type, dev.index, child_k.dtype, leaf is not None, win2 is not None)
    ok = _CANARY.get(key)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        ok = _CANARY[key] = _canary(dev, child_k.dtype, leaf is not None, win2 is not None)
        if not ok and not _warned:
            import warnings
            warnings.warn("recsa.mlp_pool_fused_v2: the bind-time canary found the two fused pools differ for "
                          f"{key}; recsa_mlp_pool_fused_kernel runs", RuntimeWarning, stacklevel=3)
            _warned = True
    return ok
