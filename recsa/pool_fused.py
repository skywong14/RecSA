"""Fused shared_std / mlp_pooler probe pooling for the prefill pyramid build."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from recsa.mlp_pool_exact import (CERT_TORCH, CERT_TRITON, FLT_LOWEST_C, _warp_tree, clamp_min_nan, div_rn, expf_nf,
                                silu_f32, sqrt_rn)

POOL_FUSED = 1
_LAUNCH = dict(enable_fp_fusion=False)
P_MIN = 64


@triton.jit
def _dtile():
    return tl.reshape(tl.arange(0, 32), (32, 1)) + 32 * tl.reshape(tl.arange(0, 4), (1, 4))


@triton.jit
def _ldx(ptr, base, j, ROW: tl.constexpr, d, nv):
    return tl.load(ptr + base + j * ROW + d, mask=(d >= 0) & (j < nv), other=0.0).to(tl.float32)


@triton.jit
def _r4_lanes(acc):
    # [32 x, 4 i] -> [1, 32]: t_x = ((acc0 + acc1) + acc2) + acc3
    a, b = tl.split(tl.reshape(acc, (32, 2, 2)))    # i0 = 0 (i = 0, 2) / 1 (i = 1, 3)   -> [32, 2 (i1)]
    e0, e2 = tl.split(a)
    e1, e3 = tl.split(b)
    return tl.reshape(((e0 + e1) + e2) + e3, (1, 32))


@triton.jit
def pool_std_seg_a(ck, mu_ptr, dev_ptr, sig_ptr, n_child, eps2, H: tl.constexpr):
    pid = tl.program_id(0)
    p = pid // H
    h = pid % H
    ROW: tl.constexpr = H * 128
    d = _dtile()
    base = p.to(tl.int64) * 16 * ROW + h * 128
    nv = tl.minimum(n_child - p * 16, 16)
    cnt = tl.maximum(nv, 1).to(tl.float32)
    den = (tl.maximum(nv, 1) * 128).to(tl.float32)
    a0 = tl.zeros((32, 4), tl.float32)
    a1 = tl.zeros((32, 4), tl.float32)
    a2 = tl.zeros((32, 4), tl.float32)
    a3 = tl.zeros((32, 4), tl.float32)
    for m in tl.static_range(4):
        a0 = a0 + _ldx(ck, base, 4 * m + 0, ROW, d, nv)
        a1 = a1 + _ldx(ck, base, 4 * m + 1, ROW, d, nv)
        a2 = a2 + _ldx(ck, base, 4 * m + 2, ROW, d, nv)
        a3 = a3 + _ldx(ck, base, 4 * m + 3, ROW, d, nv)
    mu = div_rn(((a0 + a1) + a2) + a3, cnt)
    tl.store(mu_ptr + (p.to(tl.int64) * H + h) * 128 + d, mu)
    acc = tl.zeros((32, 4), tl.float32)
    dbase = p.to(tl.int64) * 16 * ROW + h * 128
    for j in tl.static_range(16):
        dv = tl.where(j < nv, _ldx(ck, base, j, ROW, d, nv) - mu, 0.0)
        tl.store(dev_ptr + dbase + j * ROW + d, dv)
        acc = acc + dv * dv
    var = div_rn(_warp_tree(_r4_lanes(acc), 1), den)                              # [1, 1]
    # sigma = var.clamp_min(eps^2).sqrt()
    tl.store(sig_ptr + p * H + h + tl.zeros([1, 1], tl.int32), sqrt_rn(clamp_min_nan(var, eps2)))


@triton.jit
def _logit(mv_ptr, p, h, j, H: tl.constexpr, tau, sig, nv):
    # logits = tau * mv / sigma, masked_fill(~valid, -inf)
    lv = div_rn(tau * tl.load(mv_ptr + (p.to(tl.int64) * 16 + j) * H + h), sig)
    return tl.where(j < nv, lv, float("-inf"))


@triton.jit
def pool_std_seg_b(ck, cv, mv_ptr, mu_ptr, sig_ptr, kout, vout, n_child, tau_ptr, g_ptr, h_ptr, eps2,
                   H: tl.constexpr, HAS_G: tl.constexpr, MLP: tl.constexpr):
    pid = tl.program_id(0)
    p = pid // H
    h = pid % H
    ROW: tl.constexpr = H * 128
    d = _dtile()
    base = p.to(tl.int64) * 16 * ROW + h * 128
    nv = tl.minimum(n_child - p * 16, 16)
    cnt = tl.maximum(nv, 1).to(tl.float32)
    sig = tl.load(sig_ptr + p * H + h)
    tau = tl.load(tau_ptr)
    # softmax over the 16 children
    mx = FLT_LOWEST_C
    for j in tl.static_range(16):
        lj = _logit(mv_ptr, p, h, j, H, tau, sig, nv)
        mx = tl.where(mx < lj, lj, mx)
    sm = 0.0
    for j in tl.static_range(16):
        sm = sm + expf_nf(_logit(mv_ptr, p, h, j, H, tau, sig, nv) - mx)
    a0 = tl.zeros((32, 4), tl.float32)
    a1 = tl.zeros((32, 4), tl.float32)
    a2 = tl.zeros((32, 4), tl.float32)
    a3 = tl.zeros((32, 4), tl.float32)
    v0 = tl.zeros((32, 4), tl.float32)
    v1 = tl.zeros((32, 4), tl.float32)
    v2 = tl.zeros((32, 4), tl.float32)
    v3 = tl.zeros((32, 4), tl.float32)
    for m in tl.static_range(4):
        al = div_rn(expf_nf(_logit(mv_ptr, p, h, 4 * m + 0, H, tau, sig, nv) - mx), sm)
        a0 = a0 + al * _ldx(ck, base, 4 * m + 0, ROW, d, nv)
        al = div_rn(expf_nf(_logit(mv_ptr, p, h, 4 * m + 1, H, tau, sig, nv) - mx), sm)
        a1 = a1 + al * _ldx(ck, base, 4 * m + 1, ROW, d, nv)
        al = div_rn(expf_nf(_logit(mv_ptr, p, h, 4 * m + 2, H, tau, sig, nv) - mx), sm)
        a2 = a2 + al * _ldx(ck, base, 4 * m + 2, ROW, d, nv)
        al = div_rn(expf_nf(_logit(mv_ptr, p, h, 4 * m + 3, H, tau, sig, nv) - mx), sm)
        a3 = a3 + al * _ldx(ck, base, 4 * m + 3, ROW, d, nv)
        v0 = v0 + _ldx(cv, base, 4 * m + 0, ROW, d, nv)
        v1 = v1 + _ldx(cv, base, 4 * m + 1, ROW, d, nv)
        v2 = v2 + _ldx(cv, base, 4 * m + 2, ROW, d, nv)
        v3 = v3 + _ldx(cv, base, 4 * m + 3, ROW, d, nv)
    pooled = ((a0 + a1) + a2) + a3
    o = (p.to(tl.int64) * H + h) * 128 + d
    mu = tl.load(mu_ptr + o)
    # k_parent = mu + g * (pooled - mu); .to(tree dtype)
    diff = pooled - mu
    if HAS_G:
        kp = mu + tl.load(g_ptr) * diff
    else:
        kp = mu + 1.0 * diff
    if MLP:
        tl.store(kout + o, kp)
        q = mu * mu
        rm = _warp_tree(_r4_lanes(0.0 + q), 1) * 0.0078125                          # [1, 1]
        rho = sqrt_rn(clamp_min_nan(rm, eps2))
        hb = h_ptr + (p.to(tl.int64) * H + h) * 256 + d
        tl.store(hb, div_rn(mu, rho))
        tl.store(hb + 128, div_rn(diff, sig))
    else:
        tl.store(kout + o, kp.to(kout.dtype.element_ty))
    vp = div_rn(((v0 + v1) + v2) + v3, cnt)
    tl.store(vout + o, vp.to(vout.dtype.element_ty))


@triton.jit
def pool_mlp_seg_c(z1_ptr, b1_ptr, hid_ptr, n, H1: tl.constexpr, BLOCK: tl.constexpr):
    # hid = silu(z1 + b1)
    o = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    x = tl.load(z1_ptr + o, mask=m, other=0.0) + tl.load(b1_ptr + o % H1, mask=m, other=0.0)
    tl.store(hid_ptr + o, silu_f32(x), mask=m)


@triton.jit
def pool_mlp_seg_d(kp0_ptr, sig_ptr, z2_ptr, kout, n, BLOCK: tl.constexpr):
    # k_parent = kp0 + sigma * z2; .to(tree dtype)
    o = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    k = tl.load(kp0_ptr + o, mask=m, other=0.0) + tl.load(sig_ptr + o // 128, mask=m, other=0.0) * tl.load(
        z2_ptr + o, mask=m, other=0.0)
    tl.store(kout + o, k.to(kout.dtype.element_ty), mask=m)


def fused_ok(child_k, child_v, c, probe_w, probe_tau, probe_gate, n_valid):
    if not (POOL_FUSED and child_k.is_cuda and child_k.dtype == torch.bfloat16 and child_v.dtype == torch.bfloat16
            and child_k.dim() == 4 and child_k.shape[0] == 1 and child_k.shape[-1] == 128
            and child_v.shape == child_k.shape and c == 16 and n_valid == child_k.shape[1]
            and -(-child_k.shape[1] // 16) >= P_MIN and child_k.is_contiguous() and child_v.is_contiguous()):
        return False
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad
                                       for t in (child_k, child_v, probe_w, probe_tau, probe_gate)):
        return False
    if probe_w.dim() != 1 or probe_tau.dim() != 0 or (probe_gate is not None and probe_gate.dim() != 0):
        return False
    return torch.__version__ == CERT_TORCH and triton.__version__ == CERT_TRITON


def pool_std_fused(child_k, child_v, probe_w, probe_tau, probe_gate, sigma_eps, debug=False):
    """Fused _pool_one_level(standardize=True); returns (k_parent, v_parent)."""
    _, n_child, h_kv, k_dim = child_k.shape
    P = -(-n_child // 16)
    dev = torch.empty((1, P, 16, h_kv, k_dim), dtype=torch.float32, device=child_k.device)
    mu = torch.empty((1, P, h_kv, k_dim), dtype=torch.float32, device=child_k.device)
    sig = torch.empty((1, P, h_kv), dtype=torch.float32, device=child_k.device)
    eps2 = float(torch.tensor(sigma_eps * sigma_eps, dtype=torch.float32))   # clamp_min's Scalar -> float
    pool_std_seg_a[(P * h_kv,)](child_k, mu, dev, sig, n_child, eps2, H=h_kv, num_warps=1, **_LAUNCH)
    mv = dev @ probe_w.to(torch.float32)
    kout = torch.empty((1, P, h_kv, k_dim), dtype=child_k.dtype, device=child_k.device)
    vout = torch.empty((1, P, h_kv, k_dim), dtype=child_v.dtype, device=child_v.device)
    tau = probe_tau.to(torch.float32)
    g = probe_gate.to(torch.float32) if probe_gate is not None else tau
    pool_std_seg_b[(P * h_kv,)](child_k, child_v, mv, mu, sig, kout, vout, n_child, tau, g, mu, eps2, H=h_kv,
                                HAS_G=probe_gate is not None, MLP=False, num_warps=1, **_LAUNCH)
    if debug:
        return kout, vout, dict(mu=mu, dev=dev, sigma=sig, mv=mv)
    return kout, vout


def pool_mlp_fused(child_k, child_v, probe_w, probe_tau, probe_gate, probe_extra, sigma_eps, debug=False):
    """Fused _pool_one_level_mlp(probe_mode="mlp_pooler"); returns (k_parent, v_parent)."""
    _, n_child, h_kv, k_dim = child_k.shape
    P = -(-n_child // 16)
    f32 = torch.float32
    dev = torch.empty((1, P, 16, h_kv, k_dim), dtype=f32, device=child_k.device)
    mu = torch.empty((1, P, h_kv, k_dim), dtype=f32, device=child_k.device)
    sig = torch.empty((1, P, h_kv), dtype=f32, device=child_k.device)
    eps2 = float(torch.tensor(sigma_eps * sigma_eps, dtype=f32))
    pool_std_seg_a[(P * h_kv,)](child_k, mu, dev, sig, n_child, eps2, H=h_kv, num_warps=1, **_LAUNCH)
    mv = dev @ probe_w.to(f32)
    kp0 = torch.empty((1, P, h_kv, k_dim), dtype=f32, device=child_k.device)
    hcat = torch.empty((1, P, h_kv, 2 * k_dim), dtype=f32, device=child_k.device)
    vout = torch.empty((1, P, h_kv, k_dim), dtype=child_v.dtype, device=child_v.device)
    tau = probe_tau.to(f32)
    g = probe_gate.to(f32) if probe_gate is not None else tau
    pool_std_seg_b[(P * h_kv,)](child_k, child_v, mv, mu, sig, kp0, vout, n_child, tau, g, hcat, eps2, H=h_kv,
                                HAS_G=probe_gate is not None, MLP=True, num_warps=1, **_LAUNCH)
    w1 = probe_extra["mlp_w1"].to(f32)
    b1 = probe_extra["mlp_b1"].to(f32)
    w2 = probe_extra["mlp_w2"].to(f32)
    z1 = hcat @ w1.transpose(0, 1)
    h1 = w1.shape[0]
    hid = torch.empty(z1.shape, dtype=f32, device=z1.device)
    n1 = z1.numel()
    pool_mlp_seg_c[(triton.cdiv(n1, 1024),)](z1, b1.contiguous(), hid, n1, H1=h1, BLOCK=1024, num_warps=4, **_LAUNCH)
    z2 = hid @ w2.transpose(0, 1)
    kout = torch.empty((1, P, h_kv, k_dim), dtype=child_k.dtype, device=child_k.device)
    n2 = kout.numel()
    pool_mlp_seg_d[(triton.cdiv(n2, 1024),)](kp0, sig, z2, kout, n2, BLOCK=1024, num_warps=4, **_LAUNCH)
    if debug:
        return kout, vout, dict(mu=mu, dev=dev, sigma=sig, mv=mv, kp0=kp0, h=hcat, z1=z1, hid=hid, z2=z2)
    return kout, vout
