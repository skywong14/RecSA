"""Finalize (rank-select) step of the fast top selection as a hand-written warp-per-row CUDA kernel (recsa/cuda/top_finalize.cu)."""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import torch

from recsa.leaf_cuda import NVCC, _driver, _launch

SRC = Path(__file__).resolve().parent / "cuda" / "top_finalize.cu"
_mods = {}


EXTRA = ()


def _module(extra=None):
    extra = EXTRA if extra is None else tuple(extra)
    dev = (torch.cuda.current_device(), extra)
    if dev in _mods:
        return _mods[dev]
    text = SRC.read_bytes()
    hsh = hashlib.sha1(text + " ".join(extra).encode()).hexdigest()[:16]
    cache = SRC.parent / ".cubin_cache"
    cache.mkdir(exist_ok=True)
    cub = cache / f"top_finalize_{hsh}.cubin"
    if not cub.exists():
        tmp = cub.with_suffix(f".tmp{os.getpid()}")
        r = subprocess.run([NVCC, "-arch=sm_90a", "-cubin", "-O3", "-lineinfo", *extra, "-o", str(tmp), str(SRC)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed: {r.stderr}")
        os.replace(tmp, cub)
    cu = _driver()
    mod = ctypes.c_void_p()
    data = cub.read_bytes()
    buf = ctypes.create_string_buffer(data, len(data))
    assert cu.cuModuleLoadData(ctypes.byref(mod), buf) == 0, "cuModuleLoadData"
    fns = {}
    for nr in (1, 2, 3, 4, 6, 8, 12, 16):
        fn = ctypes.c_void_p()
        assert cu.cuModuleGetFunction(ctypes.byref(fn), mod, f"top_finalize_nr{nr}".encode()) == 0
        fns[nr] = fn
    _mods[dev] = (fns, mod, buf)
    return _mods[dev]


def build():
    _module()


NR_SET = (1, 2, 3, 4, 6, 8, 12, 16)


def top_finalize(raw, pmax, next_sel, bitmask, *, t_base, q0, T, QC, NP, NT, NWORDS, lq0, nq, nr, B_Hkv, layer_power,
                 H_kv, TOP_K, LOG_N, svt, neg_inf):
    """One top_finalize launch: queries lq0 .. lq0 + nq of the chunk, row count nr (one of NR_SET)."""
    fns = _module()[0]
    _launch(fns[nr], ((nq + 3) // 4, B_Hkv, 1), (128, 1, 1), 0,
            [raw, pmax, next_sel, bitmask, int(t_base), int(q0), int(T), int(QC), int(NP), int(NT), int(NWORDS),
             int(lq0), int(nq), int(layer_power), int(H_kv), int(TOP_K), int(LOG_N), float(svt), float(neg_inf)])


def mid_finalize(raw, pmax, psel, nsel, *, t_offset, t_base, n_prog_t, T, B_Hkv, layer_power, H_kv, LOG_N, svt,
                 neg_inf):
    """mid_select_finalize_kernel for TOP_K = 256, CR = 16: one warp per (query, kv head) row."""
    mod = _module()
    if "mid" not in mod[0]:
        for nr, name in ((16, b"mid_finalize16"), (12, b"mid_finalize_nr12"), (8, b"mid_finalize_nr8"),
                         (4, b"mid_finalize_nr4")):
            fn = ctypes.c_void_p()
            assert _driver().cuModuleGetFunction(ctypes.byref(fn), mod[1], name) == 0
            mod[0][("mid", nr)] = fn
        mod[0]["mid"] = mod[0][("mid", 16)]
    lp_par = layer_power * 16
    s0 = 0
    while s0 < n_prog_t:
        nr, s1 = 16, n_prog_t
        bound = min(256, (t_base + t_offset + s0) // lp_par + 1)
        nr = next(x for x in (4, 8, 12, 16) if 16 * x >= bound)
        if nr < 16:
            s1 = min(n_prog_t, 16 * nr * lp_par - t_base - t_offset)    # first query with bound > 16 nr
        _launch(mod[0][("mid", nr)], ((s1 - s0 + 3) // 4, B_Hkv, 1), (128, 1, 1), 0,
                [raw[s0 * 4096:], pmax[s0 * 256:], psel, nsel, int(t_offset + s0), int(t_base), int(n_prog_t),
                 int(s1 - s0), int(T), int(layer_power), int(H_kv), int(LOG_N), float(svt), float(neg_inf)])
        s0 = s1


ASRC = Path(__file__).resolve().parent / "cuda" / "top_score.cu"
ANST = 2                            # ring stages
AFLAGS = ("-DMINB=3", "-DSACC=1")
_amods = {}


def asmem_bytes(nst, extra=()):
    rsa = "-DRSA=0" not in extra
    return (0 if rsa else 2 * 16384) + nst * 32768 + 8 * nst + 1024


def _amodule(nst=None, extra=None):
    nst = ANST if nst is None else nst
    extra = (f"-DNST={nst}",) + (AFLAGS if extra is None else tuple(extra))
    key = (torch.cuda.current_device(), extra)
    if key in _amods:
        return _amods[key][0]
    text = ASRC.read_bytes()
    hsh = hashlib.sha1(text + " ".join(extra).encode()).hexdigest()[:16]
    cache = ASRC.parent / ".cubin_cache"
    cache.mkdir(exist_ok=True)
    cub = cache / f"top_score_{hsh}.cubin"
    if not cub.exists():
        tmp = cub.with_suffix(f".tmp{os.getpid()}")
        r = subprocess.run([NVCC, "-arch=sm_90a", "-cubin", "-O3", "-lineinfo", *extra, "-o", str(tmp), str(ASRC)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed: {r.stderr}")
        os.replace(tmp, cub)
    cu = _driver()
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    data = cub.read_bytes()
    buf = ctypes.create_string_buffer(data, len(data))
    assert cu.cuModuleLoadData(ctypes.byref(mod), buf) == 0, "cuModuleLoadData"
    assert cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"top_score_cuda") == 0, "cuModuleGetFunction"
    assert cu.cuFuncSetAttribute(fn, 8, asmem_bytes(nst, extra)) == 0, "cuFuncSetAttribute(max dynamic smem)"
    _amods[key] = (fn, mod, buf)
    return fn


def krot_map(krot):
    """3-D TMA map over the pre-split [2, B, N, H_kv, 128] fp16 keys: dims (128, H_kv, 2 B N), box (64, 1, 64)."""
    from recsa.leaf_cuda import TensorMap
    two, B, N, Hkv, K = krot.shape
    assert two == 2 and krot.dtype == torch.float16 and K == 128 and krot.is_contiguous() and krot.data_ptr() % 16 == 0
    m = TensorMap()
    dims = (ctypes.c_uint64 * 3)(K, Hkv, 2 * B * N)
    strides = (ctypes.c_uint64 * 2)(K * 2, Hkv * K * 2)
    box = (ctypes.c_uint32 * 3)(64, 1, 64)
    estr = (ctypes.c_uint32 * 3)(1, 1, 1)
    r = _driver().cuTensorMapEncodeTiled(ctypes.byref(m), 6, 3, ctypes.c_void_p(krot.data_ptr()), dims, strides, box,
                                         estr, 0, 3, 2, 0)
    assert r == 0, f"cuTensorMapEncodeTiled failed: {r}"
    return m


def top_score(q, krot, raw, pmax, cos_cache, sin_cache, *, t_base, q0, qc, T, QC, N_layer, NP, NT, layer_power, scale,
              neg_inf, kmap=None, nst=None, extra=None):
    """fast_select.top_select_score_kernel (PREC 5, QB 16, G 4, K 128, TILE_N 64) for queries q0 .. q0 + qc."""
    B, Tq, H, K = q.shape
    Hkv = krot.shape[3]
    assert K == 128 and H == 4 * Hkv and q.dtype == torch.bfloat16 and q.is_contiguous()
    assert cos_cache.dtype == torch.float32 and cos_cache.is_contiguous() and sin_cache.is_contiguous()
    nst = ANST if nst is None else nst
    fn = _amodule(nst, extra)
    _launch(fn, ((qc + 15) // 16, B * Hkv, 1), (128, 1, 1), asmem_bytes(nst, AFLAGS if extra is None else extra),
            [q, cos_cache, sin_cache, raw, pmax, int(t_base), int(q0), int(T), int(QC), int(N_layer), int(NP), int(NT),
             int(layer_power), H, Hkv, B, float(scale), float(neg_inf), kmap if kmap is not None else krot_map(krot)])


MSRC = Path(__file__).resolve().parent / "cuda" / "mid_score.cu"
MNS = 6                             # ring slots
MFLAGS = ("-DMINB=3", "-DCSM=1")
_mmods = {}


def msmem_bytes(ns):
    return ns * 8192 + 8 * ns + 64 + 512 + 1024


def _mmodule(ns=None, extra=None):
    ns = MNS if ns is None else ns
    extra = (f"-DNS={ns}",) + (MFLAGS if extra is None else tuple(extra))
    key = (torch.cuda.current_device(), extra)
    if key in _mmods:
        return _mmods[key][0]
    text = MSRC.read_bytes()
    hsh = hashlib.sha1(text + " ".join(extra).encode()).hexdigest()[:16]
    cache = MSRC.parent / ".cubin_cache"
    cache.mkdir(exist_ok=True)
    cub = cache / f"mid_score_{hsh}.cubin"
    if not cub.exists():
        tmp = cub.with_suffix(f".tmp{os.getpid()}")
        r = subprocess.run([NVCC, "-arch=sm_90a", "-cubin", "-O3", "-lineinfo", *extra, "-o", str(tmp), str(MSRC)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed: {r.stderr}")
        os.replace(tmp, cub)
    cu = _driver()
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    data = cub.read_bytes()
    buf = ctypes.create_string_buffer(data, len(data))
    assert cu.cuModuleLoadData(ctypes.byref(mod), buf) == 0, "cuModuleLoadData"
    assert cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"mid_score_cuda") == 0, "cuModuleGetFunction"
    assert cu.cuFuncSetAttribute(fn, 8, msmem_bytes(ns)) == 0, "cuFuncSetAttribute(max dynamic smem)"
    _mmods[key] = (fn, mod, buf)
    return fn


def krot_map16(krot):
    """3-D TMA map over the pre-split [2, B, N, H_kv, 128] fp16 keys: dims (128, H_kv, 2 B N), box (64, 1, 16)."""
    from recsa.leaf_cuda import TensorMap
    two, B, N, Hkv, K = krot.shape
    assert two == 2 and krot.dtype == torch.float16 and K == 128 and krot.is_contiguous() and krot.data_ptr() % 16 == 0
    m = TensorMap()
    dims = (ctypes.c_uint64 * 3)(K, Hkv, 2 * B * N)
    strides = (ctypes.c_uint64 * 2)(K * 2, Hkv * K * 2)
    box = (ctypes.c_uint32 * 3)(64, 1, 16)
    estr = (ctypes.c_uint32 * 3)(1, 1, 1)
    r = _driver().cuTensorMapEncodeTiled(ctypes.byref(m), 6, 3, ctypes.c_void_p(krot.data_ptr()), dims, strides, box,
                                         estr, 0, 3, 2, 0)
    assert r == 0, f"cuTensorMapEncodeTiled failed: {r}"
    return m


def mid_score(q, krot, ulist, umask, ucount, posq, cos_cache, sin_cache, raw, pmax, *, t_offset, t_base, n_prog_t,
              N_layer, layer_power, scale, neg_inf, kmap=None, ns=None, extra=None):
    """fast_select.mid_select_union_kernel (PREC 3, BQ 16, G 4, K 128, CR 16, TOP_K 256) for one chunk."""
    B, T, H, K = q.shape
    Hkv = krot.shape[3]
    assert K == 128 and H == 4 * Hkv and q.dtype == torch.bfloat16 and q.is_contiguous()
    for t in (ulist, umask, ucount, posq, cos_cache, sin_cache):
        assert t.is_contiguous()
    ns = MNS if ns is None else ns
    fn = _mmodule(ns, extra)
    _launch(fn, ((n_prog_t + 15) // 16, B * Hkv, 1), (128, 1, 1), msmem_bytes(ns),
            [q, ulist, umask, ucount, posq, cos_cache, sin_cache, raw, pmax, int(t_offset), int(t_base), int(n_prog_t),
             T, int(N_layer), ulist.shape[-1], int(layer_power), H, Hkv, B, float(scale), float(neg_inf),
             kmap if kmap is not None else krot_map16(krot)])
