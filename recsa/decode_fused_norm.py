"""One-launch RMSNorm and silu*mul for the decode step's model side."""
from __future__ import annotations

import struct
import warnings

import torch
import triton
import triton.language as tl

from recsa.mlp_pool_exact import silu_f32

CERT_TORCH = "2.7.1+cu128"
CERT_TRITON = "3.3.1"
CERT_REDUCE_CUH_SHA16 = "afde4f3f5831bec6"
MIRRORED_TRANSFORMERS = "5.9.0"
CERT_SM = (9, 0)

HIDDEN = 2560                      # Qwen3-4B hidden size
HEAD = 128                         # head dim
ROWS128 = (8, 32)                  # k_norm (8 kv heads), q_norm (32 q heads)
F2560 = struct.unpack("f", struct.pack("f", 1.0 / 2560.0))[0]  # fp32(1)/fp32(2560), RNE
_LAUNCH = dict(enable_fp_fusion=False)
_WARPS = dict(m2560=4, k8=8, q32=8, silu=8)
_SILU_BLOCK = 512


@triton.jit
def rsqrt_nf(a):
    return tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=r,r", [a],
                                     dtype=tl.float32, is_pure=True, pack=1)


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


# ---------------------------------------------------------------- kernels

@triton.jit
def recsa_decode_rmsnorm2560_kernel(
    x_ptr,     # bf16 [2560] (contiguous)
    w_ptr,     # bf16 [2560]
    out_ptr,   # bf16 [2560] OUT
    dbg_ptr,
    EPS: tl.constexpr,      # fp32-representable eps
    FACTOR: tl.constexpr,   # fp32(1/2560)
    DBG: tl.constexpr = False,
):
    # tile [512 lanes, 4 vector slots]: element 4x + i (and 2048 + 4x + i)
    r = tl.arange(0, 512)[:, None]
    i = tl.arange(0, 4)[None, :]
    oa = 4 * r + i
    mc = (r < 128) & (i < 4)
    xa = tl.load(x_ptr + oa).to(tl.float32)
    xc = tl.load(x_ptr + 2048 + oa, mask=mc, other=0.0).to(tl.float32)
    v = (0.0 + xa * xa) + xc * xc                         # [512, 4]
    ev, od = tl.split(tl.reshape(v, (512, 2, 2)))         # i = 2*hi + lo: lo = 0 / 1
    v0, v2 = tl.split(ev)
    v1, v3 = tl.split(od)
    t = tl.reshape(((v0 + v1) + v2) + v3, (1, 512))
    t = _halve(t, 1, 512)
    t = _halve(t, 1, 256)
    t = _halve(t, 1, 128)
    t = _halve(t, 1, 64)
    var = _warp_tree(t, 1) * FACTOR                       # [1, 1]
    rs = rsqrt_nf(var + EPS)
    ya = xa * rs
    yc = xc * rs
    wa = tl.load(w_ptr + oa).to(tl.float32)
    wc = tl.load(w_ptr + 2048 + oa, mask=mc, other=0.0).to(tl.float32)
    tl.store(out_ptr + oa, (wa * ya.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16))
    tl.store(out_ptr + 2048 + oa, (wc * yc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16),
             mask=mc)
    if DBG:
        tl.store(dbg_ptr + tl.zeros([1, 1], tl.int32), var)
        tl.store(dbg_ptr + 1 + oa, ya)
        tl.store(dbg_ptr + 1 + 2048 + oa, yc, mask=mc)


@triton.jit
def recsa_decode_rmsnorm128_kernel(
    x_ptr,     # bf16 [R, 128] (rows 128 apart)
    w_ptr,     # bf16 [128]
    out_ptr,   # bf16 [R, 128] OUT
    dbg_ptr,   # fp32 [R + R * 128] OUT (DBG only): var | y
    EPS: tl.constexpr,
    R: tl.constexpr,
    DBG: tl.constexpr = False,
):
    rr = tl.arange(0, R)
    if R == 8:
        # lane x in [0, 64) holds q_x and q_{x+64}
        o = rr[:, None] * 128 + tl.arange(0, 64)[None, :]
        a = tl.load(x_ptr + o).to(tl.float32)
        c = tl.load(x_ptr + 64 + o).to(tl.float32)
        t = (((0.0 + a * a) + (0.0 + c * c)) + 0.0) + 0.0
        s = _warp_tree(_halve(t, R, 64), R)               # [R, 1]
    else:
        # lane x in [0, 32) holds q_x, q_{x+32}, q_{x+64}, q_{x+96}
        o = rr[:, None] * 128 + tl.arange(0, 32)[None, :]
        a0 = tl.load(x_ptr + o).to(tl.float32)
        a1 = tl.load(x_ptr + 32 + o).to(tl.float32)
        a2 = tl.load(x_ptr + 64 + o).to(tl.float32)
        a3 = tl.load(x_ptr + 96 + o).to(tl.float32)
        t = (((0.0 + a0 * a0) + (0.0 + a1 * a1)) + (0.0 + a2 * a2)) + (0.0 + a3 * a3)
        s = _warp_tree(t, R)                              # [R, 1]
    var = s * 0.0078125                                   # MeanOps factor 2^-7
    rs = rsqrt_nf(var + EPS)                              # [R, 1]
    of = rr[:, None] * 128 + tl.arange(0, 128)[None, :]
    y = tl.load(x_ptr + of).to(tl.float32) * rs
    w = tl.load(w_ptr + tl.arange(0, 128)).to(tl.float32)
    tl.store(out_ptr + of, (w[None, :] * y.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16))
    if DBG:
        tl.store(dbg_ptr + rr[:, None], var)
        tl.store(dbg_ptr + R + of, y)


@triton.jit
def recsa_decode_silu_mul_kernel(
    g_ptr,     # bf16 [N] gate_proj(x)
    u_ptr,     # bf16 [N] up_proj(x)
    out_ptr,   # bf16 [N] OUT: bf16(silu(gate)) * up
    N,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
    a = silu_f32(g).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + offs, (a * u).to(tl.bfloat16), mask=m)


# ---------------------------------------------------------------- launchers

def eps_f32(eps: float) -> float:
    """eps rounded to fp32 (nearest even), as torch adds it in `var + eps`."""
    return struct.unpack("f", struct.pack("f", float(eps)))[0]


def _kind(x: torch.Tensor):
    """'2560' / 8 / 32 for the supported decode shapes, else None."""
    if x.dim() == 3 and tuple(x.shape) == (1, 1, HIDDEN) and x.stride(-1) == 1:
        return "2560"
    if (x.dim() == 4 and x.shape[0] == 1 and x.shape[1] == 1 and x.shape[2] in ROWS128
            and x.shape[3] == HEAD and x.stride(-1) == 1 and x.stride(-2) == HEAD):
        return int(x.shape[2])
    return None


def rmsnorm_fused(x: torch.Tensor, weight: torch.Tensor, eps: float,
                  dbg: torch.Tensor = None) -> torch.Tensor:
    """Qwen3RMSNorm(weight, eps)(x) in one launch, for bf16 x [1,1,2560] or [1,1,R,128] (R = 8, 32) and a bf16 [W]
    weight."""
    k = _kind(x)
    W = x.shape[-1]
    if (k is None or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or weight.dim() != 1 or weight.shape[0] != W or not weight.is_contiguous()
            or not x.is_cuda or weight.device != x.device):
        raise ValueError(f"rmsnorm_fused: unsupported x {tuple(x.shape)} {x.dtype} "
                         f"stride {x.stride()} / weight {tuple(weight.shape)} {weight.dtype}")
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    on = dbg is not None
    if on:
        R = 1 if k == "2560" else k
        if dbg.dtype != torch.float32 or dbg.numel() < R * (1 + W) or not dbg.is_contiguous():
            raise ValueError("rmsnorm_fused: dbg must be contiguous fp32 [R * (1 + W)]")
    d = dbg if on else out
    e = eps_f32(eps)
    with torch.cuda.device(x.device):
        if k == "2560":
            recsa_decode_rmsnorm2560_kernel[(1,)](x, weight, out, d, EPS=e, FACTOR=F2560, DBG=on,
                                                  num_warps=_WARPS["m2560"], **_LAUNCH)
        else:
            recsa_decode_rmsnorm128_kernel[(1,)](x, weight, out, d, EPS=e, R=k, DBG=on,
                                                 num_warps=_WARPS["k8" if k == 8 else "q32"], **_LAUNCH)
    return out


def silu_mul_fused(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """F.silu(gate) * up in one launch, for bf16 gate / up of one shape with a contiguous last dim."""
    if (gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16 or gate.shape != up.shape
            or not gate.is_cuda or up.device != gate.device
            or not _flat_ok(gate) or not _flat_ok(up)):
        raise ValueError(f"silu_mul_fused: unsupported gate {tuple(gate.shape)} {gate.dtype} "
                         f"stride {gate.stride()} / up {tuple(up.shape)} {up.dtype} stride {up.stride()}")
    out = torch.empty(gate.shape, dtype=torch.bfloat16, device=gate.device)
    N = gate.numel()
    with torch.cuda.device(gate.device):
        recsa_decode_silu_mul_kernel[(triton.cdiv(N, _SILU_BLOCK),)](
            gate, up, out, N, BLOCK=_SILU_BLOCK, num_warps=_WARPS["silu"], **_LAUNCH)
    return out


def _flat_ok(t: torch.Tensor) -> bool:
    return t.dim() >= 1 and t.stride(-1) == 1 and all(s == 1 for s in t.shape[:-1])


# ---------------------------------------------------------------- dispatch

_warned = False
_MIRROR = None
CALLS = {"norm": 0, "silu": 0}


_ARCH_OK = {}    # CUDA device index -> compute capability == CERT_SM


def build_ok(device=None) -> bool:
    """True when torch / Triton and the compute capability of `device` match the builds the reduction orders were
    derived for; warns once otherwise."""
    global _warned
    why = None
    if torch.__version__ != CERT_TORCH or triton.__version__ != CERT_TRITON:
        why = (f"torch {torch.__version__} / triton {triton.__version__} are not the certified "
               f"{CERT_TORCH} / {CERT_TRITON}")
    else:
        idx = getattr(device, "index", device)
        idx = torch.cuda.current_device() if idx is None else int(idx)
        ok = _ARCH_OK.get(idx)
        if ok is None:
            ok = _ARCH_OK[idx] = tuple(torch.cuda.get_device_capability(idx)) == CERT_SM
        if not ok:
            why = (f"compute capability {tuple(torch.cuda.get_device_capability(idx))} of cuda:{idx} "
                   f"is not the certified {CERT_SM}")
    if why is not None and not _warned:
        warnings.warn(
            f"recsa.decode_fused_norm: {why}; decode runs the HF RMSNorm / MLP modules — re-derive "
            "the reduction orders before re-enabling", RuntimeWarning, stacklevel=3)
        _warned = True
    return why is None


def _mirrored():
    global _MIRROR
    if _MIRROR is None:
        try:
            import transformers
            from transformers.activations import SiLUActivation
            from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm
        except Exception:  # noqa: BLE001
            _MIRROR = False
        else:
            if transformers.__version__ == MIRRORED_TRANSFORMERS:
                _MIRROR = (Qwen3RMSNorm.forward, Qwen3MLP.forward, SiLUActivation.forward)
            else:
                warnings.warn(
                    f"recsa.decode_fused_norm: transformers {transformers.__version__} != mirrored "
                    f"{MIRRORED_TRANSFORMERS}; decode runs the HF RMSNorm / MLP modules",
                    RuntimeWarning, stacklevel=3)
                _MIRROR = False
    return _MIRROR


def _plain(mod) -> bool:
    """mod(...) == type(mod).forward(mod, ...): no instance override, no hooks."""
    return not (mod._forward_hooks or mod._forward_pre_hooks or "forward" in mod.__dict__)


def _global_hooks_free() -> bool:
    from torch.nn.modules import module as _nnm
    return not (getattr(_nnm, "_global_forward_hooks", None)
                or getattr(_nnm, "_global_forward_pre_hooks", None))


def _no_autograd_no_autocast(*ts) -> bool:
    if torch.is_autocast_enabled("cuda"):
        return False
    return not (torch.is_grad_enabled() and any(t.requires_grad for t in ts))


def norm_eligible(mod, x: torch.Tensor) -> bool:
    """True when rmsnorm_fused(x, mod.weight, mod.variance_epsilon) can replace mod(x)."""
    mir = _mirrored()
    if not mir or type(mod).forward is not mir[0] or not _plain(mod):
        return False
    w = mod.weight
    return (_kind(x) is not None and x.dtype == torch.bfloat16 and x.is_cuda
            and isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 1
            and w.shape[0] == x.shape[-1] and w.is_contiguous() and w.device == x.device
            and isinstance(mod.variance_epsilon, float)
            # fp32(eps) a nonzero denormal would reach Triton as an fp64 constexpr
            and (eps_f32(mod.variance_epsilon) == 0.0 or abs(eps_f32(mod.variance_epsilon)) >= 2.0 ** -126)
            and _global_hooks_free() and _no_autograd_no_autocast(x, w) and build_ok(x.device))


def _mlp_call_is_mirrored(mlp, x: torch.Tensor, fwd) -> bool:
    """True when mlp(x) runs `fwd` (the mirrored Qwen3MLP.forward), with no hooks and no instance forward other than
    the chunked-MLP wrapper."""
    if mlp._forward_hooks or mlp._forward_pre_hooks:
        return False
    inst = mlp.__dict__.get("forward")
    if inst is None:
        return True
    orig = getattr(mlp, "_recsa_mlp_orig", None)
    chunk = getattr(mlp, "_recsa_mlp_chunk", None)
    return (getattr(getattr(inst, "__func__", None), "_recsa_chunked_mlp", False) is True
            and getattr(inst, "__self__", None) is mlp
            and getattr(orig, "__func__", None) is fwd and getattr(orig, "__self__", None) is mlp
            and (not chunk or x.shape[1] <= chunk))


def mlp_eligible(mlp, x: torch.Tensor) -> bool:
    """True when mlp(x) can run as down_proj(silu_mul_fused(gate_proj(x), up_proj(x))) for one decode row."""
    mir = _mirrored()
    if not mir or type(mlp).forward is not mir[1] or not _mlp_call_is_mirrored(mlp, x, mir[1]):
        return False
    act = getattr(mlp, "act_fn", None)
    return (isinstance(act, torch.nn.Module) and type(act).forward is mir[2] and _plain(act)
            and x.dtype == torch.bfloat16 and x.is_cuda and x.dim() == 3
            and x.shape[0] == 1 and x.shape[1] == 1
            and _global_hooks_free() and _no_autograd_no_autocast(x) and build_ok(x.device))


def rms_norm(mod, x: torch.Tensor, fused: bool) -> torch.Tensor:
    """mod(x) for an RMSNorm module, in one launch when fused=True and norm_eligible."""
    if fused and norm_eligible(mod, x):
        CALLS["norm"] += 1
        return rmsnorm_fused(x, mod.weight, mod.variance_epsilon)
    return mod(x)


def mlp_forward(mlp, x: torch.Tensor, fused_silu: bool) -> torch.Tensor:
    """mlp(x) for a Qwen3MLP; with fused_silu=True and mlp_eligible, silu(gate) * up runs as one launch."""
    if fused_silu and mlp_eligible(mlp, x):
        g = mlp.gate_proj(x)
        u = mlp.up_proj(x)
        if (g.dtype == torch.bfloat16 and u.dtype == torch.bfloat16 and g.shape == u.shape
                and _flat_ok(g) and _flat_ok(u) and g.device == u.device):
            CALLS["silu"] += 1
            return mlp.down_proj(silu_mul_fused(g, u))
        return mlp.down_proj(mlp.act_fn(g) * u)
    return mlp(x)
