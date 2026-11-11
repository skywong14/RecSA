"""Hand-written sm_90a leaf union walk (recsa/cuda/leaf_union.cu): build, load and launch (merge_acc.LEAF_CUDA)."""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent / "cuda" / "leaf_union.cu"
NVCC = os.environ.get("RECSA_NVCC", "/usr/local/cuda/bin/nvcc")
CRN = 16
NWARP, NSLOT, EG = 4, 8, 4
FLAGS = ("-DC_SMEM=1", "-DMINB=3", "-DGL=1")

_cu = None
_fn = {}

CUDA_STRICT = False
CUDA_FALLBACKS = []


class CudaKernelUnavailable(RuntimeError):
    """An sm_90a kernel of the fast prefill failed to build / load / launch while strict mode was on."""


def set_cuda_strict(on=True):
    global CUDA_STRICT
    CUDA_STRICT = bool(on)


def cuda_strict():
    return CUDA_STRICT or os.environ.get("RECSA_CUDA_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}


def cuda_fallback(family, ex):
    """Called in each except path of the CUDA dispatch: raises CudaKernelUnavailable under strict mode, otherwise
    records the fallback."""
    if cuda_strict():
        raise CudaKernelUnavailable(
            f"[recsa] CUDA kernel family {family!r} failed ({type(ex).__name__}: {ex}); strict mode "
            "(RECSA_CUDA_STRICT / set_cuda_strict, on in eval runs) forbids the silent fallback to the Triton path. "
            "Fix the toolchain (nvcc, driver, recsa/cuda/.cubin_cache) or opt out explicitly (RECSA_PREFILL_FAST=0)."
        ) from ex
    CUDA_FALLBACKS.append({"family": family, "error": f"{type(ex).__name__}: {str(ex)[:300]}"})


class TensorMap(ctypes.Structure):
    _fields_ = [("opaque", ctypes.c_uint64 * 16)]


def _driver():
    global _cu
    if _cu is None:
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        cu.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        cu.cuTensorMapEncodeTiled.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p,
                                                                                ctypes.POINTER(ctypes.c_void_p),
                                                                                ctypes.c_void_p]
        _cu = cu
    return _cu


def smem_bytes(nslot=NSLOT, eg=EG):
    return nslot * 8192 + (nslot // eg) * 12 + 1024 + 512


def build(extra=None, smem=None):
    """Compile (cached by source + flags hash) and load the kernel for the current device; returns the CUfunction."""
    smem = smem_bytes() if smem is None else smem
    if extra is None:
        extra = (f"-DNWARP={NWARP}", f"-DNSLOT={NSLOT}", f"-DEG={EG}", "-DLMAX=0", "-DUSE_TMA=1", *FLAGS)
    dev = torch.cuda.current_device()
    key = (tuple(extra), dev, smem)
    if key in _fn:
        return _fn[key][0]
    text = SRC.read_bytes()
    hsh = hashlib.sha1(text + " ".join(extra).encode()).hexdigest()[:16]
    cache = SRC.parent / ".cubin_cache"
    cache.mkdir(exist_ok=True)
    cub = cache / f"leaf_union_{hsh}.cubin"
    if not cub.exists():
        tmp = cub.with_suffix(f".tmp{os.getpid()}")
        r = subprocess.run([NVCC, "-arch=sm_90a", "-cubin", "-O3", "-lineinfo", *extra, "-o", str(tmp), str(SRC)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed: {r.stderr}")
        os.replace(tmp, cub)
    cu = _driver()
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    data = cub.read_bytes()
    buf = ctypes.create_string_buffer(data, len(data))
    assert cu.cuModuleLoadData(ctypes.byref(mod), buf) == 0, "cuModuleLoadData"
    assert cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"leaf_union_cuda") == 0, "cuModuleGetFunction"
    assert cu.cuFuncSetAttribute(fn, 8, smem) == 0, "cuFuncSetAttribute(max dynamic smem)"
    _fn[key] = (fn, mod, buf)
    return fn


def tensor_map(t):
    """3-D TMA map over a [B, N, H_kv, 128] fp16 tensor: dims (128, H_kv, B*N), box (64, 1, 16), 128B swizzle."""
    B, N, Hkv, K = t.shape
    assert t.dtype == torch.float16 and K == 128 and t.is_contiguous() and t.data_ptr() % 16 == 0
    m = TensorMap()
    dims = (ctypes.c_uint64 * 3)(K, Hkv, B * N)
    strides = (ctypes.c_uint64 * 2)(K * 2, Hkv * K * 2)
    box = (ctypes.c_uint32 * 3)(64, 1, CRN)
    estr = (ctypes.c_uint32 * 3)(1, 1, 1)
    r = _driver().cuTensorMapEncodeTiled(ctypes.byref(m), 6, 3, ctypes.c_void_p(t.data_ptr()), dims, strides, box,
                                         estr, 0, 3, 2, 0)
    assert r == 0, f"cuTensorMapEncodeTiled failed: {r}"
    return m


def _launch(fn, grid, block, smem, args):
    holders = []
    for a in args:
        if isinstance(a, torch.Tensor):
            holders.append(ctypes.c_void_p(a.data_ptr()))
        elif isinstance(a, (ctypes.Structure, ctypes.c_int64, ctypes.c_void_p)):
            holders.append(a)
        elif isinstance(a, float):
            holders.append(ctypes.c_float(a))
        else:
            holders.append(ctypes.c_int(int(a)))
    arr = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
    r = _driver().cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], smem,
                                 ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), arr, None)
    assert r == 0, f"cuLaunchKernel failed: {r}"


def supported(q, k_layer, v_layer, CR):
    B, T, H, K = q.shape
    return (q.is_cuda and torch.cuda.get_device_capability(q.device) == (9, 0) and q.dtype == torch.bfloat16
            and K == 128 and v_layer.shape[-1] == 128 and H == 4 * k_layer.shape[2] and CR == CRN
            and q.is_contiguous())


def leaf_union_cuda(q, krot, v16, ulist, umask, ucount, posq, cos_cache, sin_cache, out_max, out_sum, out_o, *,
                    t_base, N_layer, scale, neg_inf, nwarp=NWARP, nslot=NSLOT, eg=EG, extra=FLAGS, fuse=None):
    """Launch like leaf_union_kernel, with union lists of BQ = 4 * nwarp (fp16 values). With fuse = (global_max,
    global_sum, global_output, out_bf16), write the final normalized output instead of the level state."""
    B, T, H, K = q.shape
    Hkv = krot.shape[2]
    assert K == 128 and H == 4 * Hkv and q.dtype == torch.bfloat16 and krot.dtype == torch.float16
    assert v16.dtype == torch.float16 and v16.shape[-1] == 128
    for t in (q, krot, v16, ulist, umask, ucount, posq, cos_cache, sin_cache, out_max, out_sum, out_o):
        assert t.is_contiguous()
    smem = smem_bytes(nslot, eg)
    if "-DGL=1" in extra and (ulist.shape[-1] % 4 or umask.shape[-1] % 4 or eg != 4 or ulist.data_ptr() % 16
                              or umask.data_ptr() % 16):
        extra = tuple(f for f in extra if f != "-DGL=1")      # int4 list reads need 16-byte aligned rows
    fn = build((f"-DNWARP={nwarp}", f"-DNSLOT={nslot}", f"-DEG={eg}", "-DLMAX=0", "-DUSE_TMA=1", *extra), smem)
    BQ = 4 * nwarp
    _launch(fn, ((T + BQ - 1) // BQ, B * Hkv, 1), (32 * nwarp, 1, 1), smem,
            [q, krot, v16, ulist, umask, ucount, posq, cos_cache, sin_cache, out_max, out_sum, out_o,
             int(t_base), T, int(N_layer), ulist.shape[-1], H, Hkv, float(scale), float(neg_inf),
             tensor_map(krot), tensor_map(v16), *(fuse if fuse is not None else (out_max, out_sum, out_o, out_o)),
             int(fuse is not None)])


USRC = Path(__file__).resolve().parent / "cuda" / "union_build.cu"
_ufn = {}


UNION_M_MAX = 32768


def _union_fn(name=b"union_build16"):
    dev = (torch.cuda.current_device(), name)
    if dev in _ufn:
        return _ufn[dev][0]
    text = USRC.read_bytes()
    hsh = hashlib.sha1(text).hexdigest()[:16]
    cache = USRC.parent / ".cubin_cache"
    cache.mkdir(exist_ok=True)
    cub = cache / f"union_build_{hsh}.cubin"
    if not cub.exists():
        tmp = cub.with_suffix(f".tmp{os.getpid()}")
        r = subprocess.run([NVCC, "-arch=sm_90a", "-cubin", "-O3", "-lineinfo", "-o", str(tmp), str(USRC)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed: {r.stderr}")
        os.replace(tmp, cub)
    cu = _driver()
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    data = cub.read_bytes()
    buf = ctypes.create_string_buffer(data, len(data))
    assert cu.cuModuleLoadData(ctypes.byref(mod), buf) == 0, "cuModuleLoadData"
    assert cu.cuModuleGetFunction(ctypes.byref(fn), mod, name) == 0, "cuModuleGetFunction"
    if name == b"union_build16m":
        assert cu.cuFuncSetAttribute(fn, 8, 2 * UNION_M_MAX + 16) == 0, "cuFuncSetAttribute(max dynamic smem)"
    if name == b"union_xmask16":
        assert cu.cuFuncSetAttribute(fn, 8, XMASK_SMEM_MAX) == 0, "cuFuncSetAttribute(max dynamic smem)"
    _ufn[dev] = (fn, mod, buf)
    return fn


XMASK_SMEM_MAX = 200 * 1024


def union_xmask16(next_sel, ulist, ucount, CR, n_parents):
    """merge_acc.union_xmask(next_sel, ulist, 16, CR, ucount) for TK = 256, or None when
    the row does not fit in shared memory. n_parents: number of parents at this level (> every n // CR)."""
    B, T, Hkv, TK = next_sel.shape
    NB, U = ulist.shape[1], ulist.shape[-1]
    PM = max(1, int(n_parents))
    smem = 4 * U * 16 + ((2 * PM + 15) // 16) * 16
    if TK != 256 or CR & (CR - 1) or smem > XMASK_SMEM_MAX or next_sel.stride(3) != 1 or next_sel.stride(2) != TK:
        return None
    rows = B * NB * Hkv
    xm = torch.empty((B, NB, Hkv, U * 16), dtype=torch.int32, device=next_sel.device)
    _launch(_union_fn(b"union_xmask16"), (rows, 1, 1), (256, 1, 1), smem,
            [next_sel, ulist.contiguous(), ucount.contiguous(), xm, T, NB, Hkv, _Long(next_sel.stride(0)),
             _Long(next_sel.stride(1)), U, PM, int(CR).bit_length() - 1])
    return xm


def union_lists16(sel, pad=True, posq_t_base=None, umax=None, n_parents=None):
    """merge_acc.union_lists(sel, 16) for TK = 256 (pad=False leaves the entries past each
    row's count unwritten). posq_t_base: also return merge_acc.query_positions(sel, posq_t_base, 16, 1). n_parents:
    number of nodes the rows index (> every listed id): with it, the mask-table builder (union_build16m)."""
    B, T, Hkv, TK = sel.shape
    assert TK == 256 and sel.dtype == torch.int32 and sel.stride(3) == 1 and sel.stride(2) == TK
    NB = (T + 15) // 16
    rows = B * NB * Hkv
    U = 16 * TK if umax is None else min(16 * TK, int(umax))
    ulist = torch.empty((rows, U), dtype=torch.int32, device=sel.device)
    umask = torch.empty((rows, U), dtype=torch.int32, device=sel.device)
    ucount = torch.empty((rows,), dtype=torch.int32, device=sel.device)
    posq = torch.empty((B, T, Hkv), dtype=torch.int32, device=sel.device) if posq_t_base is not None else None
    args = [sel, ulist, umask, ucount, T, NB, Hkv, _Long(sel.stride(0)), _Long(sel.stride(1)), U, int(pad),
            posq if posq is not None else ctypes.c_void_p(0), int(posq_t_base or 0)]
    if n_parents is not None and 0 < int(n_parents) <= UNION_M_MAX:
        _launch(_union_fn(b"union_build16m"), (rows, 1, 1), (256, 1, 1), 2 * int(n_parents) + 16,
                args + [int(n_parents)])
    else:
        _launch(_union_fn(), (rows, 1, 1), (256, 1, 1), 0, args)
    out = (ulist.view(B, NB, Hkv, U), umask.view(B, NB, Hkv, U), ucount.view(B, NB, Hkv))
    return out + (posq,) if posq is not None else out


class _Long(ctypes.c_int64):
    pass


ASRC = Path(__file__).resolve().parent / "cuda" / "top_accum.cu"
_afn = {}
AFLAGS = ("-DWG=1", "-DMINB=3")


def asmem(flags):
    ns = 2
    for f in flags:
        if f.startswith("-DNSTAGE="):
            ns = int(f.split("=")[1])
    return ns * 4 * 64 * 128 + ns * 8 + 1024


def tensor_map64(t):
    """3-D TMA map over [B, N, H_kv, 128] fp16: dims (128, H_kv, B*N), box (64, 1, 64 nodes), 128B swizzle."""
    B, N, Hkv, K = t.shape
    assert t.dtype == torch.float16 and K == 128 and t.is_contiguous() and t.data_ptr() % 16 == 0
    m = TensorMap()
    dims = (ctypes.c_uint64 * 3)(K, Hkv, B * N)
    strides = (ctypes.c_uint64 * 2)(K * 2, Hkv * K * 2)
    box = (ctypes.c_uint32 * 3)(64, 1, 64)
    estr = (ctypes.c_uint32 * 3)(1, 1, 1)
    r = _driver().cuTensorMapEncodeTiled(ctypes.byref(m), 6, 3, ctypes.c_void_p(t.data_ptr()), dims, strides, box,
                                         estr, 0, 3, 2, 0)
    assert r == 0, f"cuTensorMapEncodeTiled failed: {r}"
    return m


def _accum_fn(flags=None):
    flags = AFLAGS if flags is None else tuple(flags)
    dev = (torch.cuda.current_device(), flags)
    if dev in _afn:
        return _afn[dev][0]
    text = ASRC.read_bytes()
    hsh = hashlib.sha1(text + " ".join(flags).encode()).hexdigest()[:16]
    cache = ASRC.parent / ".cubin_cache"
    cache.mkdir(exist_ok=True)
    cub = cache / f"top_accum_{hsh}.cubin"
    if not cub.exists():
        tmp = cub.with_suffix(f".tmp{os.getpid()}")
        r = subprocess.run([NVCC, "-arch=sm_90a", "-cubin", "-O3", "-lineinfo", *flags, "-o", str(tmp), str(ASRC)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed: {r.stderr}")
        os.replace(tmp, cub)
    cu = _driver()
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    data = cub.read_bytes()
    buf = ctypes.create_string_buffer(data, len(data))
    assert cu.cuModuleLoadData(ctypes.byref(mod), buf) == 0, "cuModuleLoadData"
    assert cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"top_accum_cuda") == 0, "cuModuleGetFunction"
    assert cu.cuFuncSetAttribute(fn, 8, asmem(flags)) == 0, "cuFuncSetAttribute(max dynamic smem)"
    _afn[dev] = (fn, mod, buf)
    return fn


def top_accum_cuda(q, krot, v16, cos_cache, sin_cache, bitmask, out_max, out_sum, out_o, *, t_base, layer_power,
                   N_layer, scale, neg_inf, flags=None, fuse_init=None):
    """recsa_top_accumulate_gqa_kernel (PREROT, fp16 dots, no bias) for G = 4, head dim 128. fuse_init = (max_out,
    sum_out, output_out): write the first merge's result (against the fresh accumulators) there instead."""
    flags = AFLAGS if flags is None else tuple(flags)
    B, T, H, K = q.shape
    Hkv = krot.shape[2]
    assert K == 128 and H == 4 * Hkv and q.dtype == torch.bfloat16 and krot.dtype == torch.float16
    for t in (q, krot, v16, cos_cache, sin_cache, bitmask, out_max, out_sum, out_o):
        assert t.is_contiguous()
    fn = _accum_fn(flags)
    _launch(fn, ((T + 15) // 16, B * Hkv, 1), (128, 1, 1), asmem(flags),
            [q, cos_cache, sin_cache, bitmask, out_max, out_sum, out_o, int(t_base), int(layer_power), T, H, Hkv,
             int(N_layer), bitmask.shape[-1], float(scale), float(neg_inf), tensor_map64(krot), tensor_map64(v16),
             *(fuse_init if fuse_init is not None else (out_max, out_sum, out_o)), int(fuse_init is not None)])
