"""Exact-order Triton segments of the MLP-pooler cursor pool used by the CUDA-graph decode step."""
from __future__ import annotations

import warnings

import torch
import triton
import triton.language as tl

CERT_TORCH = "2.7.1+cu128"
CERT_TRITON = "3.3.1"
CERT_REDUCE_CUH_SHA16 = "afde4f3f5831bec6"

C_EXACT, H_EXACT, D_EXACT = 16, 8, 128
FLT_LOWEST = -3.4028234663852886e38   # -FLT_MAX: SoftMax.cu's max identity
FLT_LOWEST_C = tl.constexpr(FLT_LOWEST)
_LAUNCH = dict(enable_fp_fusion=False)
_WARPS = dict(a=2, b=2, c=8, d=1)
DBG_OFF = dict(var=0, logits=1, alpha=17, pooled=33, rho=161)
DBG_SIZE = 162
_DBG_VAR = tl.constexpr(DBG_OFF["var"])
_DBG_LOGITS = tl.constexpr(DBG_OFF["logits"])
_DBG_ALPHA = tl.constexpr(DBG_OFF["alpha"])
_DBG_POOLED = tl.constexpr(DBG_OFF["pooled"])
_DBG_RHO = tl.constexpr(DBG_OFF["rho"])


@triton.jit
def div_rn(a, b):
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=r,r,r", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def sqrt_rn(a):
    return tl.inline_asm_elementwise("sqrt.rn.f32 $0, $1;", "=r,r", [a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def expf_nf(a):
    return tl.inline_asm_elementwise(
        """{
        .reg .f32 t0, t1, t2, t3, t4, t5, t6, t7;
        .reg .b32 b0, b1;
        fma.rn.f32 t0, $1, 0f3BBB989D, 0f3F000000;
        cvt.sat.f32.f32 t1, t0;
        fma.rm.f32 t2, t1, 0f437C0000, 0f4B400001;
        add.rn.f32 t3, t2, 0fCB40007F;
        neg.f32 t3, t3;
        fma.rn.f32 t4, $1, 0f3FB8AA3B, t3;
        fma.rn.f32 t5, $1, 0f32A57060, t4;
        mov.b32 b0, t2;
        shl.b32 b1, b0, 23;
        mov.b32 t6, b1;
        ex2.approx.ftz.f32 t7, t5;
        mul.rn.f32 $0, t7, t6;
        }""", "=r,r", [a], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def clamp_min_nan(v, lo):
    # torch clamp_min(Scalar), CUDA: _isnan(v) ? v : ::max(v, lower)
    return tl.where(v != v, v, tl.maximum(v, lo))


@triton.jit
def silu_f32(x):
    # torch silu, CUDA: x / (1 + expf(-x)) in fp32 opmath
    return div_rn(x, 1.0 + expf_nf(-x))


@triton.jit
def _halve(t, R: tl.constexpr, W: tl.constexpr):
    # [R, W] -> [R, W/2]: t[:, x] + t[:, x + W/2]   (block_x_reduce, shared memory)
    a, b = tl.split(tl.permute(tl.reshape(t, (R, 2, W // 2)), (0, 2, 1)))
    return a + b


@triton.jit
def _pair(t, R: tl.constexpr, W: tl.constexpr):
    # [R, W] -> [R, W/2]: t[:, 2k] + t[:, 2k+1]       (one shfl_down level)
    a, b = tl.split(tl.reshape(t, (R, W // 2, 2)))
    return a + b


@triton.jit
def _warp_tree(t, R: tl.constexpr):
    # [R, 32] -> [R, 1]: shfl_down offsets 1, 2, 4, 8, 16; lane 0's value
    t = _pair(t, R, 32)
    t = _pair(t, R, 16)
    t = _pair(t, R, 8)
    t = _pair(t, R, 4)
    t = _pair(t, R, 2)
    return t


@triton.jit
def _jtile():
    # child index j over the bucket dims (i0, i1, m1, m0): [1, 2, 2, 2, 2]
    i0 = tl.reshape(tl.arange(0, 2), (1, 2, 1, 1, 1))
    i1 = tl.reshape(tl.arange(0, 2), (1, 1, 2, 1, 1))
    m1 = tl.reshape(tl.arange(0, 2), (1, 1, 1, 2, 1))
    m0 = tl.reshape(tl.arange(0, 2), (1, 1, 1, 1, 2))
    return 8 * m1 + 4 * m0 + 2 * i1 + i0


@triton.jit
def _r1_tile(x):
    a, b = tl.split(x)          # m0 = 0 / 1                -> [64, i0, i1, m1]
    x0, x2 = tl.split(a)        # m = 0 (m1 = 0), m = 2 (m1 = 1)
    x1, x3 = tl.split(b)        # m = 1, m = 3              -> [64, i0, i1]
    acc = (((0.0 + x0) + x1) + x2) + x3
    c, e = tl.split(acc)        # i1 = 0 / 1                -> [64, i0]
    a0, a1 = tl.split(c)        # i = 0, 1
    a2, a3 = tl.split(e)        # i = 2, 3
    return ((a0 + a1) + a2) + a3


@triton.jit
def _r4_tile(q):
    a, b = tl.split(q)          # m0 = 0 / 1                -> [64, i0, i1, m1]
    a0, a1 = tl.split(a)        # (m0 = 0, m1 = 0 / 1)      -> [64, i0, i1]
    b0, b1 = tl.split(b)        # (m0 = 1, m1 = 0 / 1)
    y0, y1 = tl.split(a0)       # m' = 0, 1                 -> [64, i0]
    y2, y3 = tl.split(b0)       # m' = 2, 3
    y4, y5 = tl.split(a1)       # m' = 4, 5
    y6, y7 = tl.split(b1)       # m' = 6, 7
    acc = (((((((0.0 + y0) + y1) + y2) + y3) + y4) + y5) + y6) + y7
    return tl.split(acc)


@triton.jit
def _seq16(e):
    # e [1, i0, i1, m1, m0] -> [1]: (((0 + e_0) + e_1) + ...) + e_15, j order
    a, b = tl.split(e)          # m0
    a0, a1 = tl.split(a)        # blocks m = 0 (a0), 2 (a1) -> [1, i0, i1]
    b0, b1 = tl.split(b)        # blocks m = 1 (b0), 3 (b1)
    p, q = tl.split(a0)         # i1 = 0 / 1                -> [1, i0]
    e0, e1 = tl.split(p)
    e2, e3 = tl.split(q)
    p, q = tl.split(b0)
    e4, e5 = tl.split(p)
    e6, e7 = tl.split(q)
    p, q = tl.split(a1)
    e8, e9 = tl.split(p)
    e10, e11 = tl.split(q)
    p, q = tl.split(b1)
    e12, e13 = tl.split(p)
    e14, e15 = tl.split(q)
    s = 0.0 + e0
    s = s + e1
    s = s + e2
    s = s + e3
    s = s + e4
    s = s + e5
    s = s + e6
    s = s + e7
    s = s + e8
    s = s + e9
    s = s + e10
    s = s + e11
    s = s + e12
    s = s + e13
    s = s + e14
    s = s + e15
    return s


@triton.jit
def recsa_mlp_pool_seg_a_kernel(
    child_k,
    idx_ptr,   # int64 [16]: the window's clamped child rows
    inv_ptr,   # uint8 [16]: the window's pad mask (j >= n)
    cnt_ptr,   # fp32 [1]: float(n)
    den_ptr,   # fp32 [1]: float(n * 128)
    mu_ptr,    # fp32 [H, 128] OUT
    dev_ptr,
    sig_ptr,   # fp32 [H] OUT
    leaf_v,    # [1, cap, H, 128] level-0 V buffer (LEAF only)
    k_t,
    v_t,       # [H, 128] the new leaf V row (LEAF only)
    t_ptr,
    dbg_ptr,   # fp32 [DBG_SIZE * H] (DBG only)
    H: tl.constexpr,
    LEAF: tl.constexpr,
    DBG: tl.constexpr = False,
):
    h = tl.program_id(0)
    ROW: tl.constexpr = H * 128
    j = _jtile()                                         # [1, 2, 2, 2, 2]
    d = tl.reshape(tl.arange(0, 64), (64, 1, 1, 1, 1))
    o0 = h * 128 + d
    o1 = o0 + 64
    rows = tl.load(idx_ptr + j)
    ok = tl.load(inv_ptr + j) == 0
    x0 = tl.load(child_k + rows * ROW + o0)
    x1 = tl.load(child_k + rows * ROW + o1)
    if LEAF:
        t_row = tl.load(t_ptr)
        kt0 = tl.load(k_t + o0)
        kt1 = tl.load(k_t + o1)
        x0 = tl.where(rows == t_row, kt0, x0)
        x1 = tl.where(rows == t_row, kt1, x1)
        ol = h * 128 + tl.reshape(tl.arange(0, 64), (1, 64))
        tl.store(child_k + t_row * ROW + ol, tl.reshape(kt0, (1, 64)))
        tl.store(child_k + t_row * ROW + ol + 64, tl.reshape(kt1, (1, 64)))
        tl.store(leaf_v + t_row * ROW + ol, tl.load(v_t + ol))
        tl.store(leaf_v + t_row * ROW + ol + 64, tl.load(v_t + ol + 64))
    x0 = tl.where(ok, x0.to(tl.float32), 0.0)
    x1 = tl.where(ok, x1.to(tl.float32), 0.0)
    cnt = tl.load(cnt_ptr)
    den = tl.load(den_ptr)
    # mu = kb.sum(dim=2, keepdim=True) / cnt5                (sum order, div)
    mu0 = div_rn(_r1_tile(x0), cnt)                      # [64]
    mu1 = div_rn(_r1_tile(x1), cnt)
    # dev = (kb - mu).masked_fill_(invalid5, 0.0)
    dv0 = tl.where(ok, x0 - tl.reshape(mu0, (64, 1, 1, 1, 1)), 0.0)
    dv1 = tl.where(ok, x1 - tl.reshape(mu1, (64, 1, 1, 1, 1)), 0.0)
    tl.store(dev_ptr + j * ROW + o0, dv0)
    tl.store(dev_ptr + j * ROW + o1, dv1)
    # var = dev.pow(2).sum(dim=(2, 4)) / den3                (var order, div)
    ev0, od0 = _r4_tile(dv0 * dv0)                       # half 0: even / odd j
    ev1, od1 = _r4_tile(dv1 * dv1)                       # half 1
    t = ((ev0 + ev1) + od0) + od1                        # acc0..acc3 of the var order
    vs = _warp_tree(_halve(tl.reshape(t, (1, 64)), 1, 64), 1)    # [1, 1]
    var = div_rn(vs, den)
    lane = tl.arange(0, 64)
    tl.store(mu_ptr + h * 128 + lane, mu0)
    tl.store(mu_ptr + h * 128 + 64 + lane, mu1)
    o11 = tl.zeros([1, 1], tl.int32)
    # sigma = var.clamp_min(eps^2).sqrt(); eps^2 = fp32(1e-6 * 1e-6)
    tl.store(sig_ptr + h + o11, sqrt_rn(clamp_min_nan(var, 1e-12)))
    if DBG:
        tl.store(dbg_ptr + _DBG_VAR * H + h + o11, var)


@triton.jit
def recsa_mlp_pool_seg_b_kernel(
    child_k,
    idx_ptr,   # int64 [16]
    inv_ptr,   # uint8 [16]
    mu_ptr,    # fp32 [H, 128] (seg A)
    sig_ptr,   # fp32 [H] (seg A)
    mv_ptr,    # fp32 [16, H]: dev @ probe_w (torch)
    tau,       # fp32 host scalar: the probe's tau
    gate,
    kp0_ptr,   # fp32 [H, 128] OUT: mu + gate * (pooled - mu)
    h_ptr,
    dbg_ptr,   # fp32 [DBG_SIZE * H] (DBG only)
    H: tl.constexpr,
    DBG: tl.constexpr = False,
):
    h = tl.program_id(0)
    ROW: tl.constexpr = H * 128
    j = _jtile()
    d = tl.reshape(tl.arange(0, 64), (64, 1, 1, 1, 1))
    o0 = h * 128 + d
    rows = tl.load(idx_ptr + j)
    ok = tl.load(inv_ptr + j) == 0
    x0 = tl.where(ok, tl.load(child_k + rows * ROW + o0).to(tl.float32), 0.0)
    x1 = tl.where(ok, tl.load(child_k + rows * ROW + o0 + 64).to(tl.float32), 0.0)
    mvj = tl.load(mv_ptr + j * H + h)                    # [1, 2, 2, 2, 2]
    sig = tl.load(sig_ptr + h)
    lane = tl.arange(0, 64)
    mu0 = tl.load(mu_ptr + h * 128 + lane)
    mu1 = tl.load(mu_ptr + h * 128 + 64 + lane)
    lm = tl.where(ok, div_rn(tau * mvj, sig), float("-inf"))
    lmf = tl.reshape(tl.where(lm != lm, FLT_LOWEST_C, lm), (16,))
    mx = tl.maximum(tl.max(lmf, axis=0), FLT_LOWEST_C)
    ex = expf_nf(lm - mx)
    s = _seq16(ex)                                       # [1]
    alpha = div_rn(ex, tl.reshape(s, (1, 1, 1, 1, 1)))
    # pooled = (alpha.unsqueeze(-1) * kb).sum(dim=2)                 (sum order)
    p0 = _r1_tile(alpha * x0)                            # [64]
    p1 = _r1_tile(alpha * x1)
    d0 = p0 - mu0
    d1 = p1 - mu1
    tl.store(kp0_ptr + h * 128 + lane, mu0 + gate * d0)
    tl.store(kp0_ptr + h * 128 + 64 + lane, mu1 + gate * d1)
    # rho = mu2.pow(2).mean(dim=-1).clamp_min(eps^2).sqrt()          (mean order)
    q0 = tl.reshape(mu0 * mu0, (1, 64))
    q1 = tl.reshape(mu1 * mu1, (1, 64))
    z = tl.zeros([1, 64], tl.float32)
    tq = (((z + q0) + (z + q1)) + 0.0) + 0.0
    rmean = tl.reshape(_warp_tree(_halve(tq, 1, 64), 1), (1,)) * 0.0078125
    rho = sqrt_rn(clamp_min_nan(rmean, 1e-12))
    hb = h_ptr + h * 256 + lane
    tl.store(hb, div_rn(mu0, rho))
    tl.store(hb + 64, div_rn(mu1, rho))
    tl.store(hb + 128, div_rn(d0, sig))
    tl.store(hb + 192, div_rn(d1, sig))
    if DBG:
        tl.store(dbg_ptr + _DBG_LOGITS * H + j * H + h, lm)
        tl.store(dbg_ptr + _DBG_ALPHA * H + j * H + h, alpha)
        tl.store(dbg_ptr + _DBG_POOLED * H + h * 128 + lane, p0)
        tl.store(dbg_ptr + _DBG_POOLED * H + h * 128 + 64 + lane, p1)
        tl.store(dbg_ptr + _DBG_RHO * H + h + tl.zeros([1], tl.int32), rho)


@triton.jit
def recsa_mlp_pool_seg_c_kernel(
    z1_ptr,    # fp32 [H, H1]: h @ W1^T (torch)
    b1_ptr,    # fp32 [H1]
    hid_ptr,   # fp32 [H, H1] OUT: silu(z1 + b1) (the gemm operand)
    H: tl.constexpr,
    H1: tl.constexpr,
):
    o = tl.arange(0, H)[:, None] * H1 + tl.arange(0, H1)[None, :]
    zb = tl.load(z1_ptr + o) + tl.load(b1_ptr + tl.arange(0, H1))[None, :]
    tl.store(hid_ptr + o, silu_f32(zb))


@triton.jit
def recsa_mlp_pool_seg_d_kernel(
    kp0_ptr,   # fp32 [H, 128] (seg B)
    sig_ptr,   # fp32 [H] (seg A)
    z2_ptr,    # fp32 [H, 128]: hid @ W2^T (torch)
    out_k,
    p_ptr,     # int64 [1]: the open parent's slot
    H: tl.constexpr,
):
    h = tl.program_id(0)
    o = h * 128 + tl.arange(0, 128)[None, :]
    sig = tl.load(sig_ptr + h + tl.zeros([1, 1], tl.int32))
    # k_parent + sigma[..., None] * (hid @ W2^T), out= the tree dtype (RNE)
    k = tl.load(kp0_ptr + o) + sig * tl.load(z2_ptr + o)
    p = tl.load(p_ptr)
    tl.store(out_k + p * (H * 128) + o, k.to(out_k.dtype.element_ty))


# ---------------------------------------------------------------- host side

_warned = False


def exact_build_ok() -> bool:
    """True for the torch / Triton builds the reduction orders were derived for."""
    return torch.__version__ == CERT_TORCH and triton.__version__ == CERT_TRITON


def exact_eligible(probe_mode: str, h_kv: int, d: int, c: int, device,
                   tree_dtype: torch.dtype) -> bool:
    """True when the exact segments can replace DecodeBucketPoolMLP.cursor_k: default bucket shape, mlp_pooler probe,
    a CUDA tree in bf16 / fp16 / fp32, and supported torch / Triton builds (warns once otherwise)."""
    global _warned
    if not (probe_mode == "mlp_pooler" and (c, h_kv, d) == (C_EXACT, H_EXACT, D_EXACT)
            and torch.device(device).type == "cuda"
            and tree_dtype in (torch.bfloat16, torch.float16, torch.float32)):
        return False
    if not exact_build_ok():
        if not _warned:
            warnings.warn(
                f"recsa.mlp_pool_exact: torch {torch.__version__} / triton "
                f"{triton.__version__} are not the certified {CERT_TORCH} / "
                f"{CERT_TRITON}; the MLP cursor pool runs the torch path — "
                "re-derive the reduction orders before re-enabling",
                RuntimeWarning, stacklevel=2)
            _warned = True
        return False
    return True


class ExactScratch:
    """Persistent fp32 operands of one pool call, shared by every layer and level (decode is single-stream)."""

    __slots__ = ("mu", "dev", "sig", "kp0", "h", "hid", "mv", "sm", "dbg", "fdbg")

    def __init__(self, h_kv: int, d: int, c: int, h1: int, device):
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("mlp_pool_exact scratch must be allocated before "
                               "CUDA-graph capture (warm step)")
        f32 = dict(device=device, dtype=torch.float32)
        self.mu = torch.empty(h_kv, d, **f32)
        self.dev = torch.empty(1, 1, c, h_kv, d, **f32)
        self.sig = torch.empty(h_kv, **f32)
        self.kp0 = torch.empty(h_kv, d, **f32)
        self.h = torch.empty(1, 1, h_kv, 2 * d, **f32)
        self.hid = torch.empty(1, 1, h_kv, h1, **f32)
        self.mv = torch.empty(c, h_kv, **f32)   # fused path: its in-kernel mv hand-off
        self.sm = torch.empty(h_kv, 2 * c, **f32)   # fused path, SMAX: softmax hand-offs
        self.dbg = None
        self.fdbg = None


def pool_k_exact(scr: ExactScratch, child_k: torch.Tensor, win, probe_w: torch.Tensor,
                 tau: float, gate: float, probe_extra, out_k: torch.Tensor,
                 leaf=None) -> None:
    """The K half of the MLP-pooler cursor pool for one (layer, level), in 4 Triton + 3 torch matmul calls. On the
    level-1 call, leaf = (leaf_v, k_t, v_t, t_dev) and seg A also writes the layer's leaf rows."""
    h_kv = child_k.shape[2]
    w1, b1, w2 = probe_extra["mlp_w1"], probe_extra["mlp_b1"], probe_extra["mlp_w2"]
    inv = win.invalid4.view(torch.uint8)
    dbg = scr.dbg if scr.dbg is not None else scr.sig
    on = scr.dbg is not None
    with torch.cuda.device(child_k.device):
        if leaf is None:
            recsa_mlp_pool_seg_a_kernel[(h_kv,)](
                child_k, win.idx, inv, win.cnt4, win.den3, scr.mu, scr.dev, scr.sig,
                child_k, child_k, child_k, win.p, dbg, H=h_kv, LEAF=False, DBG=on,
                num_warps=_WARPS["a"], **_LAUNCH)
        else:
            leaf_v, k_t, v_t, t_dev = leaf
            recsa_mlp_pool_seg_a_kernel[(h_kv,)](
                child_k, win.idx, inv, win.cnt4, win.den3, scr.mu, scr.dev, scr.sig,
                leaf_v, k_t, v_t, t_dev, dbg, H=h_kv, LEAF=True, DBG=on,
                num_warps=_WARPS["a"], **_LAUNCH)
        mv = scr.dev @ probe_w                               # [1,1,16,H] (cuBLAS)
        recsa_mlp_pool_seg_b_kernel[(h_kv,)](
            child_k, win.idx, inv, scr.mu, scr.sig, mv, tau, gate, scr.kp0, scr.h, dbg,
            H=h_kv, DBG=on, num_warps=_WARPS["b"], **_LAUNCH)
        z1 = scr.h @ w1.transpose(0, 1)                      # [1,1,H,H1] (cuBLAS)
        recsa_mlp_pool_seg_c_kernel[(1,)](z1, b1, scr.hid, H=h_kv, H1=w1.shape[0],
                                          num_warps=_WARPS["c"], **_LAUNCH)
        z2 = scr.hid @ w2.transpose(0, 1)                    # [1,1,H,128] (cuBLAS)
        recsa_mlp_pool_seg_d_kernel[(h_kv,)](scr.kp0, scr.sig, z2, out_k, win.p,
                                             H=h_kv, num_warps=_WARPS["d"], **_LAUNCH)


FUSED_WARPS = 2
FUSED_LEVELS_PER_LAUNCH = 2
FUSED_SMAX = 1
FDBG_SIZE = 16 + 64 + 128
FDBG_SIZE_C = tl.constexpr(FDBG_SIZE)


@triton.jit
def fma_rn(a, b, c):
    return tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=r,r,r,r", [a, b, c],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _cols8(t, R: tl.constexpr):
    # [R, 8] -> its 8 columns [R] in column order (pure data movement)
    a, b = tl.split(tl.reshape(t, (R, 2, 2, 2)))         # col = 4*i0 + 2*i1 + i2; i2 = 0 / 1
    a0, a1 = tl.split(a)                                 # i1
    b0, b1 = tl.split(b)
    c0, c4 = tl.split(a0)                                # i0
    c2, c6 = tl.split(a1)
    c1, c5 = tl.split(b0)
    c3, c7 = tl.split(b1)
    return c0, c1, c2, c3, c4, c5, c6, c7


@triton.jit
def _mv_emul(dev_ptr, pw_ptr, mv_ptr, h, H: tl.constexpr):
    jv = tl.arange(0, 16)[:, None]
    t = tl.arange(0, 32)[None, :]
    base = dev_ptr + jv * (H * 128) + h * 128 + 4 * t                   # [16, 32]
    acc = tl.zeros([16, 32], tl.float32)
    for v in tl.static_range(4):
        acc = fma_rn(tl.load(base + v), tl.load(pw_ptr + 4 * t + v), acc)
    acc = _halve(acc, 16, 32)                            # lane x + lane x+16
    acc = _halve(acc, 16, 16)                            # + 8
    acc = _halve(acc, 16, 8)                             # + 4
    acc = _halve(acc, 16, 4)                             # + 2
    acc = _halve(acc, 16, 2)                             # + 1   -> [16, 1]
    tl.store(mv_ptr + jv * H + h, acc + 0.0)             # cuBLAS's -0 -> +0 epilogue


@triton.jit
def _z1_emul(x_ptr, w1t_ptr, H1: tl.constexpr):
    n = tl.arange(0, H1)[:, None]
    sv = tl.arange(0, 8)[None, :]
    acc = tl.zeros([H1, 8], tl.float32)
    for kl in tl.static_range(32):
        acc = fma_rn(tl.load(x_ptr + 32 * sv + kl), tl.load(w1t_ptr + (32 * sv + kl) * H1 + n), acc)
    p0, p1, p2, p3, p4, p5, p6, p7 = _cols8(acc, H1)
    return (((((((p0 + p1) + p2) + p3) + p4) + p5) + p6) + p7) + 0.0      # -0 -> +0 epilogue


@triton.jit
def _z2_chain(x_ptr, w2t_ptr, n, t: tl.constexpr):
    # chain t of z2: k = t, t+16, t+32, t+48 ascending, fma from +0
    acc = tl.zeros([128], tl.float32)
    for i in tl.static_range(4):
        acc = fma_rn(tl.load(x_ptr + 16 * i + t), tl.load(w2t_ptr + (16 * i + t) * 128 + n), acc)
    return acc


@triton.jit
def _z2_emul(x_ptr, w2t_ptr, H1: tl.constexpr):
    n = tl.arange(0, 128)
    r = _z2_chain(x_ptr, w2t_ptr, n, 0) + _z2_chain(x_ptr, w2t_ptr, n, 1)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 2)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 3)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 4)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 5)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 6)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 7)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 8)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 9)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 10)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 11)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 12)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 13)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 14)
    r = r + _z2_chain(x_ptr, w2t_ptr, n, 15)
    return r + 0.0                                       # -0 -> +0 epilogue


@triton.jit
def recsa_mlp_pool_products_kernel(
    dev_ptr,   # fp32 [16, H, 128]
    pw_ptr,    # fp32 [128]
    x1_ptr,    # fp32 [H, 256]
    w1_ptr,    # fp32 [256, H1]: W1^T
    x2_ptr,    # fp32 [H, H1]
    w2_ptr,    # fp32 [H1, 128]: W2^T
    mv_ptr,    # fp32 [16, H] OUT
    z1_ptr,    # fp32 [H, H1] OUT
    z2_ptr,    # fp32 [H, 128] OUT
    H: tl.constexpr,
    H1: tl.constexpr,
):
    """The three products alone, through the helpers the fused kernel calls."""
    h = tl.program_id(0)
    _mv_emul(dev_ptr, pw_ptr, mv_ptr, h, H)
    tl.store(z1_ptr + h * H1 + tl.arange(0, H1), _z1_emul(x1_ptr + h * 256, w1_ptr, H1))
    tl.store(z2_ptr + h * 128 + tl.arange(0, 128), _z2_emul(x2_ptr + h * H1, w2_ptr, H1))


@triton.jit
def _fused_level(child_k, idx_ptr, inv_ptr, cnt_ptr, den_ptr, pw_ptr, w1_ptr, b1_ptr, w2_ptr, tau, gate,
                 dev_ptr, sig_ptr, h_ptr, hid_ptr, kp0_ptr, mv_ptr, sm_ptr, out_k, p_ptr, leaf_v, k_t, v_t,
                 t_ptr, dbg_ptr, h, H: tl.constexpr, H1: tl.constexpr, LEAF: tl.constexpr,
                 DBG: tl.constexpr, SMAX: tl.constexpr):
    ROW: tl.constexpr = H * 128
    # ---------------------------------------------------------- seg A (verbatim)
    j = _jtile()                                         # [1, 2, 2, 2, 2]
    d = tl.reshape(tl.arange(0, 64), (64, 1, 1, 1, 1))
    o0 = h * 128 + d
    o1 = o0 + 64
    rows = tl.load(idx_ptr + j)
    ok = tl.load(inv_ptr + j) == 0
    x0 = tl.load(child_k + rows * ROW + o0)
    x1 = tl.load(child_k + rows * ROW + o1)
    if LEAF:
        t_row = tl.load(t_ptr)
        kt0 = tl.load(k_t + o0)
        kt1 = tl.load(k_t + o1)
        x0 = tl.where(rows == t_row, kt0, x0)
        x1 = tl.where(rows == t_row, kt1, x1)
        ol = h * 128 + tl.reshape(tl.arange(0, 64), (1, 64))
        tl.store(child_k + t_row * ROW + ol, tl.reshape(kt0, (1, 64)))
        tl.store(child_k + t_row * ROW + ol + 64, tl.reshape(kt1, (1, 64)))
        tl.store(leaf_v + t_row * ROW + ol, tl.load(v_t + ol))
        tl.store(leaf_v + t_row * ROW + ol + 64, tl.load(v_t + ol + 64))
    x0 = tl.where(ok, x0.to(tl.float32), 0.0)
    x1 = tl.where(ok, x1.to(tl.float32), 0.0)
    cnt = tl.load(cnt_ptr)
    den = tl.load(den_ptr)
    mu0 = div_rn(_r1_tile(x0), cnt)                      # [64]
    mu1 = div_rn(_r1_tile(x1), cnt)
    dv0 = tl.where(ok, x0 - tl.reshape(mu0, (64, 1, 1, 1, 1)), 0.0)
    dv1 = tl.where(ok, x1 - tl.reshape(mu1, (64, 1, 1, 1, 1)), 0.0)
    tl.store(dev_ptr + j * ROW + o0, dv0)
    tl.store(dev_ptr + j * ROW + o1, dv1)
    ev0, od0 = _r4_tile(dv0 * dv0)
    ev1, od1 = _r4_tile(dv1 * dv1)
    t = ((ev0 + ev1) + od0) + od1
    vs = _warp_tree(_halve(tl.reshape(t, (1, 64)), 1, 64), 1)    # [1, 1]
    var = div_rn(vs, den)
    o11 = tl.zeros([1, 1], tl.int32)
    tl.store(sig_ptr + h + o11, sqrt_rn(clamp_min_nan(var, 1e-12)))
    tl.debug_barrier()
    # ---------------------------------------------------------- mv (cuBLAS order)
    _mv_emul(dev_ptr, pw_ptr, mv_ptr, h, H)              # -> mv_ptr[j * H + h]
    tl.debug_barrier()
    # ---------------------------------------------------------- seg B (verbatim)
    mvj = tl.load(mv_ptr + j * H + h)      # [1, 2, 2, 2, 2]
    if DBG:
        tl.store(dbg_ptr + j * H + h, mvj)
    sig = tl.load(sig_ptr + h)
    lane = tl.arange(0, 64)
    if SMAX:
        jv = tl.arange(0, 16)
        okv = tl.load(inv_ptr + jv) == 0
        mvv = tl.load(mv_ptr + jv * H + h)
        lmv = tl.where(okv, div_rn(tau * mvv, sig), float("-inf"))
        mx = tl.maximum(tl.max(tl.where(lmv != lmv, FLT_LOWEST_C, lmv), axis=0), FLT_LOWEST_C)
        exv = expf_nf(lmv - mx)
        tl.store(sm_ptr + h * 32 + jv, exv)
        tl.debug_barrier()
        s = _seq16(tl.load(sm_ptr + h * 32 + j))     # [1]
        tl.store(sm_ptr + h * 32 + 16 + jv, div_rn(exv, s))
        tl.debug_barrier()
        alpha = tl.load(sm_ptr + h * 32 + 16 + j)
    else:
        lm = tl.where(ok, div_rn(tau * mvj, sig), float("-inf"))
        lmf = tl.reshape(tl.where(lm != lm, FLT_LOWEST_C, lm), (16,))
        mx = tl.maximum(tl.max(lmf, axis=0), FLT_LOWEST_C)
        ex = expf_nf(lm - mx)
        s = _seq16(ex)                                   # [1]
        alpha = div_rn(ex, tl.reshape(s, (1, 1, 1, 1, 1)))
    p0 = _r1_tile(alpha * x0)                            # [64]
    p1 = _r1_tile(alpha * x1)
    d0 = p0 - mu0
    d1 = p1 - mu1
    tl.store(kp0_ptr + h * 128 + lane, mu0 + gate * d0)
    tl.store(kp0_ptr + h * 128 + 64 + lane, mu1 + gate * d1)
    q0 = tl.reshape(mu0 * mu0, (1, 64))
    q1 = tl.reshape(mu1 * mu1, (1, 64))
    z = tl.zeros([1, 64], tl.float32)
    tq = (((z + q0) + (z + q1)) + 0.0) + 0.0
    rmean = tl.reshape(_warp_tree(_halve(tq, 1, 64), 1), (1,)) * 0.0078125
    rho = sqrt_rn(clamp_min_nan(rmean, 1e-12))
    hb = h_ptr + h * 256 + lane
    tl.store(hb, div_rn(mu0, rho))
    tl.store(hb + 64, div_rn(mu1, rho))
    tl.store(hb + 128, div_rn(d0, sig))
    tl.store(hb + 192, div_rn(d1, sig))
    tl.debug_barrier()
    n1 = tl.arange(0, H1)
    z1 = _z1_emul(h_ptr + h * 256, w1_ptr, H1)           # [H1]
    if DBG:
        tl.store(dbg_ptr + 16 * H + h * H1 + n1, z1)
    zb = z1 + tl.load(b1_ptr + n1)
    tl.store(hid_ptr + h * H1 + n1, silu_f32(zb))
    tl.debug_barrier()
    n2 = tl.arange(0, 128)
    z2 = _z2_emul(hid_ptr + h * H1, w2_ptr, H1)          # [128]
    if DBG:
        tl.store(dbg_ptr + 16 * H + H * H1 + h * 128 + n2, z2)
    k = tl.load(kp0_ptr + h * 128 + n2) + sig * z2
    p = tl.load(p_ptr)
    tl.store(out_k + p * (H * 128) + h * 128 + n2, k.to(out_k.dtype.element_ty))


@triton.jit
def recsa_mlp_pool_fused_kernel(
    child_k,
    idx_ptr,   # int64 [16]: the window's clamped child rows
    inv_ptr,   # uint8 [16]: the window's pad mask (j >= n)
    cnt_ptr,   # fp32 [1]: float(n)
    den_ptr,   # fp32 [1]: float(n * 128)
    pw_ptr,    # fp32 [128]: probe_w
    w1_ptr,    # fp32 [256, H1]: mlp_w1^T (fused_weights)
    b1_ptr,    # fp32 [H1]: mlp_b1
    w2_ptr,    # fp32 [H1, 128]: mlp_w2^T (fused_weights)
    tau,       # fp32 host scalar
    gate,      # fp32 host scalar
    dev_ptr,   # fp32 [16, H, 128] scratch (this head's slice)
    sig_ptr,   # fp32 [H] scratch
    h_ptr,     # fp32 [H, 256] scratch
    hid_ptr,   # fp32 [H, H1] scratch
    kp0_ptr,   # fp32 [H, 128] scratch
    mv_ptr,    # fp32 [16, H] scratch
    sm_ptr,    # fp32 [H, 32] scratch (SMAX only)
    out_k,
    p_ptr,     # int64 [1]: the open parent's slot
    leaf_v,    # [1, cap, H, 128] level-0 V buffer (LEAF only)
    k_t,       # [H, 128] the new leaf K row (LEAF only)
    v_t,       # [H, 128] the new leaf V row (LEAF only)
    t_ptr,
    idx2_ptr,
    inv2_ptr,
    cnt2_ptr,
    den2_ptr,
    out2_k,
    p2_ptr,
    dbg_ptr,
    H: tl.constexpr,
    H1: tl.constexpr,
    LEAF: tl.constexpr,
    TWO: tl.constexpr = False,
    DBG: tl.constexpr = False,
    SMAX: tl.constexpr = 0,
):
    """grid (H,)."""
    h = tl.program_id(0)
    _fused_level(child_k, idx_ptr, inv_ptr, cnt_ptr, den_ptr, pw_ptr, w1_ptr, b1_ptr, w2_ptr, tau, gate,
                 dev_ptr, sig_ptr, h_ptr, hid_ptr, kp0_ptr, mv_ptr, sm_ptr, out_k, p_ptr, leaf_v, k_t, v_t,
                 t_ptr, dbg_ptr, h, H, H1, LEAF, DBG, SMAX)
    if TWO:
        tl.debug_barrier()
        _fused_level(out_k, idx2_ptr, inv2_ptr, cnt2_ptr, den2_ptr, pw_ptr, w1_ptr, b1_ptr, w2_ptr,
                     tau, gate, dev_ptr, sig_ptr, h_ptr, hid_ptr, kp0_ptr, mv_ptr, sm_ptr, out2_k, p2_ptr,
                     leaf_v, k_t, v_t, t_ptr, dbg_ptr + FDBG_SIZE_C * H, h, H, H1, False, DBG, SMAX)


def fused_weights(probe_extra):
    """(W1^T [256, 64], W2^T [64, 128]) as contiguous fp32 copies, the fused kernel's weight layout. Made once per state
    at bind."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("mlp_pool_exact.fused_weights must run before CUDA-graph capture")
    return (probe_extra["mlp_w1"].t().contiguous(), probe_extra["mlp_w2"].t().contiguous())


def pool_k_fused(scr: ExactScratch, child_k: torch.Tensor, win, probe_w: torch.Tensor,
                 tau: float, gate: float, probe_extra, out_k: torch.Tensor,
                 leaf=None, win2=None, out2_k=None, wt=None) -> None:
    """pool_k_exact as one launch, on the exact path's scratch buffers."""
    from recsa import mlp_pool_fused_v2 as _v2
    if _v2.RECSA_MLP_POOL_FUSED2 == "on" and _v2.v2_ok(child_k, out_k, leaf, win2, out2_k):
        _v2.pool_k_fused_v2(scr, child_k, win, probe_w, tau, gate, probe_extra, out_k, leaf, win2, out2_k, wt)
        return
    _pool_k_fused_v1(scr, child_k, win, probe_w, tau, gate, probe_extra, out_k, leaf, win2, out2_k, wt)


def _pool_k_fused_v1(scr: ExactScratch, child_k: torch.Tensor, win, probe_w: torch.Tensor,
                     tau: float, gate: float, probe_extra, out_k: torch.Tensor,
                     leaf=None, win2=None, out2_k=None, wt=None) -> None:
    """pool_k_fused's recsa_mlp_pool_fused_kernel launch, whatever RECSA_MLP_POOL_FUSED2 says."""
    h_kv = child_k.shape[2]
    b1 = probe_extra["mlp_b1"]
    w1, w2 = fused_weights(probe_extra) if wt is None else wt
    inv = win.invalid4.view(torch.uint8)
    dbg = scr.fdbg if scr.fdbg is not None else scr.sig
    on = scr.fdbg is not None
    two = win2 is not None
    if two != (out2_k is not None):
        raise ValueError("pool_k_fused: win2 and out2_k go together")
    w2_ = win2 if two else win
    o2 = out2_k if two else out_k
    if leaf is None:
        leaf_v, k_t, v_t, t_dev = child_k, child_k, child_k, win.p
    else:
        leaf_v, k_t, v_t, t_dev = leaf
    with torch.cuda.device(child_k.device):
        recsa_mlp_pool_fused_kernel[(h_kv,)](
            child_k, win.idx, inv, win.cnt4, win.den3, probe_w, w1, b1, w2, tau, gate,
            scr.dev, scr.sig, scr.h, scr.hid, scr.kp0, scr.mv, scr.sm, out_k, win.p,
            leaf_v, k_t, v_t, t_dev,
            w2_.idx, w2_.invalid4.view(torch.uint8), w2_.cnt4, w2_.den3, o2, w2_.p,
            dbg, H=h_kv, H1=w1.shape[1], LEAF=leaf is not None, TWO=two, DBG=on,
            SMAX=FUSED_SMAX, num_warps=FUSED_WARPS, **_LAUNCH)


def products_fused(dev: torch.Tensor, probe_w: torch.Tensor, h: torch.Tensor, w1: torch.Tensor,
                   hid: torch.Tensor, w2: torch.Tensor):
    """(mv, z1, z2) from the fused kernel's helpers, shaped like dev @ probe_w, h @ w1.T, hid @ w2.T."""
    h_kv = dev.shape[3]
    h1 = w1.shape[0]
    w1, w2 = w1.t().contiguous(), w2.t().contiguous()
    mv = torch.empty(1, 1, C_EXACT, h_kv, device=dev.device, dtype=torch.float32)
    z1 = torch.empty(1, 1, h_kv, h1, device=dev.device, dtype=torch.float32)
    z2 = torch.empty(1, 1, h_kv, D_EXACT, device=dev.device, dtype=torch.float32)
    with torch.cuda.device(dev.device):
        recsa_mlp_pool_products_kernel[(h_kv,)](dev, probe_w, h, w1, hid, w2, mv, z1, z2,
                                                H=h_kv, H1=h1, num_warps=FUSED_WARPS, **_LAUNCH)
    return mv, z1, z2


CERT_SM = (9, 0)
CERT_CUBLAS = "12.8.3"
CERT_CUBLAS_KERNELS = {
    "mv": ["std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, float, float, float, "
           "float, false, true, true, false, 7, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched"
           "<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, "
           "float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, "
           "cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)"],
    "z1": ["void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>"
           "(cutlass_80_simt_sgemm_64x64_8x5_tn_align1::Params)",
           "void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, "
           "float, true, false, false>(cublasLt::cublasSplitKParams<float>, float const*, float const*, float*, "
           "float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*, "
           "float*, float*, float const*, float const*, float const*, float const*, float const*)"],
    "z2": ["void gemmSN_TN_kernel<float, 128, 16, 2, 4, 8, 9, false, cublasGemvTensorStridedBatched<float "
           "const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >"
           "(cublasGemmSmallNParams<cublasGemvTensorStridedBatched<float const>, "
           "cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)"],
}
_FUSED_CANARY: dict = {}
_fused_warned = False


def _canary_fill(scr: "ExactScratch", fam: int, g: torch.Generator) -> None:
    for t in (scr.dev, scr.h, scr.hid):
        x = torch.randn(t.shape, generator=g, device=t.device)
        if fam == 1:      # wide magnitudes, mixed signs
            x = x * torch.pow(10.0, torch.rand(t.shape, generator=g, device=t.device) * 12 - 6)
        elif fam == 2:    # cancellation
            x = torch.where(torch.rand(t.shape, generator=g, device=t.device) < 0.5, x, -x) * 1e3 + x * 1e-3
        elif fam == 3:    # exact (signed) zeros among small values
            z = torch.where(torch.rand(t.shape, generator=g, device=t.device) < 0.5, 0.0, -0.0)
            x = torch.where(torch.rand(t.shape, generator=g, device=t.device) < 0.5, z, x * 1e-3)
        t.copy_(x)


def fused_canary(probe_w: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, device) -> bool:
    """Check that the fused path's emulated products equal torch's cuBLAS products on 4 seeded input families."""
    dev = torch.device(device)
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())

    key = (dev.type, dev.index, probe_w.data_ptr(), w1.data_ptr(), w2.data_ptr())
    if key in _FUSED_CANARY:
        return _FUSED_CANARY[key]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("mlp_pool_exact.fused_canary must run before CUDA-graph capture")
    h_kv = H_EXACT
    scr = ExactScratch(h_kv, D_EXACT, C_EXACT, w1.shape[0], dev)
    g = torch.Generator(device=dev)
    g.manual_seed(20261011)
    ok = True
    with torch.cuda.device(dev):
        for fam in range(4):
            _canary_fill(scr, fam, g)
            ref = (scr.dev @ probe_w, scr.h @ w1.transpose(0, 1), scr.hid @ w2.transpose(0, 1))
            got = products_fused(scr.dev, probe_w, scr.h, w1, scr.hid, w2)
            for r, o in zip(ref, got):
                ok = ok and r.shape == o.shape and torch.equal(r.view(torch.int32), o.view(torch.int32))
    _FUSED_CANARY[key] = ok
    return ok


def fused_eligible(probe_w, probe_extra, device) -> bool:
    """True when the fused launch can replace pool_k_exact for this state: default MLP-pooler parameter shapes (fp32,
    contiguous), supported torch / Triton / compute capability, TF32 matmul off, and a passing fused_canary."""

    global _fused_warned
    if probe_w is None or probe_extra is None:
        return False
    w1, b1, w2 = (probe_extra.get("mlp_w1"), probe_extra.get("mlp_b1"),
                  probe_extra.get("mlp_w2"))
    shapes = ((probe_w, (D_EXACT,)), (w1, (64, 2 * D_EXACT)), (b1, (64,)), (w2, (D_EXACT, 64)))
    dev = torch.device(device)
    if dev.type != "cuda":
        return False
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if any(t is None or tuple(t.shape) != s or t.dtype != torch.float32 or not t.is_contiguous()
           or t.device != dev for t, s in shapes):
        return False
    device = dev
    why = None
    if not exact_build_ok():
        why = f"torch {torch.__version__} / triton {triton.__version__} are not the certified builds"
    elif torch.cuda.get_device_capability(device) != CERT_SM:
        why = f"compute capability {torch.cuda.get_device_capability(device)} != {CERT_SM}"
    elif torch.backends.cuda.matmul.allow_tf32:
        why = "torch.backends.cuda.matmul.allow_tf32 is on (the orders are the fp32 kernels')"
    elif not fused_canary(probe_w, w1, w2, device):
        why = ("the bind-time canary found the torch/cuBLAS products NOT bit-identical to the "
               "emulated orders (cuBLAS kernel or order changed)")
    if why is not None:
        if not _fused_warned:
            warnings.warn(f"recsa.mlp_pool_exact: fused MLP pool disabled ({why}); the exact "
                          "segments run instead — re-derive the FUSED ORDERS before "
                          "re-enabling", RuntimeWarning, stacklevel=2)
            _fused_warned = True
        return False
    return True
