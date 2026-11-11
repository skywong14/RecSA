"""The fast evaluation prefill (RECSA_PREFILL_FAST) against the reference evaluation prefill."""
from __future__ import annotations

import argparse
import collections
import functools
import json
import math
import os
import sys
import time
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

import recsa.fast_select as FS  # noqa: E402
import recsa.leaf_cuda as LC  # noqa: E402
import recsa.merge_acc as MA  # noqa: E402
import recsa.parallel as P  # noqa: E402
import recsa.pool_fused as PF  # noqa: E402
import recsa.topsel_cuda as TS  # noqa: E402
from recsa.reference.oracles import (RecSAParams, fixed_selection_forward_from_parents,  # noqa: E402
                                     init_prev_selected_parents)
from recsa.parallel import recsa_forward  # noqa: E402
from recsa.probe_pool import _pool_one_level, _pool_one_level_mlp  # noqa: E402
from tests.conftest import pin_deployed_prefill  # noqa: E402

pytestmark = pytest.mark.gpu

CR, TK, MTN = 16, 256, 4096
HQ, HKV, D = 32, 8, 128
G = HQ // HKV
ROPE = 1e6
SEG = 64
LOWF = tuple(range(48, 64)) + tuple(range(112, 128))
SIG_Q, SIG_K, A_K, B_Q, V_SCALE = 0.5, 0.5, 7.0, 7.0, 0.5
CASES = {"nl2": 32768, "nl3": 66048}
TEST_SEED = 0
MEASURE_SEEDS = (0, 1, 2, 3, 4, 5)
N_ORACLE_ROWS = 24
EVIDENCE = "fast_tests_evidence.json"


def _pin_fast(mp):
    """RECSA_PREFILL_FAST=1 with every other dispatch global at its fast-prefill default."""
    for name, val in (("ACC_DOT_MODE", 1), ("ACC_FP16", True), ("MERGE_ACC", True), ("MERGE_MAX_NODES", 131072),
                      ("UNION_LEAF", True), ("UNION_MID", True), ("FAST_SELECT", True), ("FAST_TOP_SELECT", None),
                      ("TOPACC_CUDA", True), ("SEL_IEEE", False)):
        mp.setattr(P, name, val)
    for name, val in (("LEAF_CUDA", 1), ("_LEAF_CUDA_BROKEN", False)):
        mp.setattr(MA, name, val)
    for name, val in (("FAST_TOP_B_CUDA", 1), ("FAST_MIDA_CUDA", 1), ("FAST_TOPA_CUDA", 1)):
        mp.setattr(FS, name, val)


def _pin_no_cuda(mp):
    """Force the Triton fallback (merge_acc.LEAF_CUDA, parallel.TOPACC_CUDA and the fast_select CUDA flags off)."""
    mp.setattr(MA, "LEAF_CUDA", 0)
    mp.setattr(P, "TOPACC_CUDA", False)
    for name in ("FAST_TOP_B_CUDA", "FAST_MIDA_CUDA", "FAST_TOPA_CUDA"):
        mp.setattr(FS, name, 0)


def _simulated_build_failure(*args, **kwargs):
    raise RuntimeError("nvcc failed: simulated by tests/test_prefill_fast.py")


def _pin_build_failure(mp):
    """Make every CUDA builder / loader raise, as on a host without nvcc, so the dispatch falls back."""
    for mod, name in ((LC, "build"), (LC, "_union_fn"), (LC, "_accum_fn"), (TS, "build"), (TS, "_amodule"),
                      (TS, "_mmodule")):
        mp.setattr(mod, name, _simulated_build_failure)
    mp.setattr(FS, "_TB", None)                          # the loaded-module cache of recsa.topsel_cuda


def _pins_of(name):
    def kill(**kv):
        def pin(mp):
            for k, v in kv.items():
                mp.setattr(P, k, v)
        return pin
    return {
        "deployed": [pin_deployed_prefill],
        "fast": [_pin_fast],
        "fast_triton": [_pin_fast, _pin_no_cuda],
        "fast_buildfail": [_pin_fast, _pin_build_failure],
        "fast_noselect": [_pin_fast, kill(FAST_SELECT=False)],
        "acc_fp16": [_pin_fast, kill(MERGE_ACC=False, UNION_LEAF=False, UNION_MID=False, FAST_SELECT=False)],
    }[name]


class _Taps:
    """Counts the prefill kernels a forward dispatches (`routes`) and keeps each level's selection
    (`sel`: N_layer -> int16 [1, T, H_kv, TK])."""

    def __init__(self, mp):
        self.routes = collections.Counter()
        self.sel = {}
        for name in ("recsa_bottom_accumulate_gqa_kernel", "recsa_top_accumulate_gqa_kernel",
                     "recsa_mid_accumulate_kernel"):
            mp.setattr(P, name, self._kernel(getattr(P, name), name))
        mp.setattr(P, "recsa_top_select_kernel", self._kernel(P.recsa_top_select_kernel, "recsa_top_select_kernel",
                                                              out_pos=4))
        mp.setattr(P, "recsa_mid_select_kernel", self._kernel(P.recsa_mid_select_kernel, "recsa_mid_select_kernel",
                                                              out_pos=5, t0_pos=8))
        mp.setattr(FS, "fast_top_select", self._fn(FS.fast_top_select, "fast_top_select", out_pos=4))
        mp.setattr(FS, "fast_mid_select_union", self._fn(FS.fast_mid_select_union, "fast_mid_select_union",
                                                         out_pos=5, chunked=True))
        for name in ("leaf_union", "mid_acc_union"):
            mp.setattr(MA, name, self._fn(getattr(MA, name), name))
        for name in ("leaf_union_cuda", "top_accum_cuda", "union_lists16", "union_xmask16"):
            mp.setattr(LC, name, self._fn(getattr(LC, name), "cuda:" + name))
        for name in ("top_score", "top_finalize", "mid_score", "mid_finalize"):
            mp.setattr(TS, name, self._fn(getattr(TS, name), "cuda:" + name))

    def _keep(self, out, n_layer):
        self.sel[int(n_layer)] = out.to(torch.int16)        # node ids < 32768 at these shapes

    def _kernel(self, kern, name, out_pos=None, t0_pos=None):
        taps = self

        class _Launch:
            def __getitem__(self, grid):
                launch = kern[grid]

                def run(*args, **kw):
                    res = launch(*args, **kw)
                    taps.routes[name] += 1
                    if out_pos is not None:
                        T = args[0].shape[1]
                        if t0_pos is None or int(args[t0_pos]) + int(grid[0]) >= T:   # last chunk of the level
                            taps._keep(args[out_pos], kw["N_layer"])
                    return res
                return run
        return _Launch()

    def _fn(self, fn, name, out_pos=None, chunked=False):
        @functools.wraps(fn)
        def run(*args, **kw):
            res = fn(*args, **kw)
            self.routes[name] += 1
            if out_pos is not None and not (name == "fast_top_select" and res is None):
                if not chunked or int(kw["t_offset"]) + int(kw["n_prog_t"]) >= args[0].shape[1]:
                    self._keep(args[out_pos], kw["N_layer"])
            return res
        return run


class _Run:
    __slots__ = ("out", "sel", "routes", "flags_after", "wall_s")


def _level_sizes(T):
    sizes = [T]
    while sizes[-1] > MTN:
        sizes.append(-(-sizes[-1] // CR))
    return sizes


def run_config(cfg, q, k, v, *, no_summaries):
    """One eval prefill under configuration `cfg` (monkeypatched, undone before returning)."""
    r = _Run()
    with pytest.MonkeyPatch.context() as mp:
        for pin in _pins_of(cfg):
            pin(mp)
        taps = _Taps(mp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = recsa_forward(q, k, v, compression_rate=CR, max_top_nodes=MTN, top_k_per_layer=TK,
                                rope_base=ROPE, rope_mode="internal", no_summaries=no_summaries)
        torch.cuda.synchronize()
        r.wall_s = time.perf_counter() - t0
        r.flags_after = {"_LEAF_CUDA_BROKEN": bool(MA._LEAF_CUDA_BROKEN), "FAST_TOP_B_CUDA": int(FS.FAST_TOP_B_CUDA),
                         "FAST_TOPA_CUDA": int(FS.FAST_TOPA_CUDA), "FAST_MIDA_CUDA": int(FS.FAST_MIDA_CUDA)}
    sizes = _level_sizes(q.shape[1])
    by_level = {}
    for lvl in range(1, len(sizes)):
        assert sizes[lvl] in taps.sel, f"{cfg}: no selection captured at level {lvl} (N={sizes[lvl]})"
        by_level[lvl] = taps.sel[sizes[lvl]]
    r.out, r.sel, r.routes = out, by_level, dict(taps.routes)
    return r


@functools.lru_cache(maxsize=None)
def cuda_side_unavailable():
    """None when the fast prefill's sm_90a kernels build and load here, else the reason (cached per process)."""
    torch.zeros(1, device="cuda")
    if torch.cuda.get_device_capability() != (9, 0):
        return f"the CUDA kernels are sm_90a only (device capability {torch.cuda.get_device_capability()})"
    try:
        LC.build()
        for name in (b"union_build16", b"union_build16m", b"union_xmask16"):
            LC._union_fn(name)
        LC._accum_fn()
        TS.build()
        TS._amodule()
        TS._mmodule()
    except Exception as ex:      # noqa: BLE001
        return f"CUDA kernels unavailable ({type(ex).__name__}: {str(ex)[:200]})"
    return None


def make_inputs(T, seed, sig_q=None, sig_k=None, a_k=None, b_q=None, v_scale=None):
    sig_q = SIG_Q if sig_q is None else sig_q
    sig_k = SIG_K if sig_k is None else sig_k
    a_k = A_K if a_k is None else a_k
    b_q = B_Q if b_q is None else b_q
    v_scale = V_SCALE if v_scale is None else v_scale
    g = torch.Generator(device="cuda").manual_seed(seed)
    f32 = dict(device="cuda", dtype=torch.float32)
    lowf = torch.tensor(LOWF, device="cuda")
    nseg = -(-T // SEG)
    topic = torch.randn(1, nseg, HKV, len(LOWF), generator=g, **f32)
    topic = topic / topic.norm(dim=-1, keepdim=True)
    vseg = torch.randn(1, nseg, HKV, D, generator=g, **f32)
    q = sig_q * torch.randn(1, T, HQ, D, generator=g, **f32)
    k = sig_k * torch.randn(1, T, HKV, D, generator=g, **f32)
    v = 0.5 * torch.randn(1, T, HKV, D, generator=g, **f32)
    seg_of_t = torch.arange(T, device="cuda") // SEG
    v = v_scale * (v + vseg[:, seg_of_t])
    k[..., lowf] += a_k * topic[:, seg_of_t]
    u = torch.rand(1, T, HKV, 2, generator=g, **f32)
    own = seg_of_t.view(1, T, 1, 1)
    needle = torch.minimum((u * (own + 1)).long(), own)                         # causal: an earlier or own segment
    hidx = torch.arange(HKV, device="cuda").view(1, HKV, 1).expand(T, HKV, 2)
    qt = topic[0][needle[0], hidx].sum(2)                                      # [T, H_kv, 32]
    q[..., lowf] += b_q * qt.repeat_interleave(G, dim=1).unsqueeze(0)
    bf = torch.bfloat16
    return q.to(bf), k.to(bf), v.to(bf), needle


def out_metrics(a, b):
    """Distance of output a from reference b ([1, T, H, V] bf16): element and row-max percentiles, fraction of rows
    above 1e-2 / 3e-2 / 1e-1, and relative L2."""
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    rowmax = d.amax(-1).flatten()
    sub = d.flatten()[::61].contiguous()
    qs = torch.tensor([0.5, 0.99, 0.999], device=d.device)
    pe = torch.quantile(sub, qs).tolist()
    pr = torch.quantile(rowmax, qs).tolist()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "p50_elem": pe[0], "p99_elem": pe[1],
            "p999_elem": pe[2], "rowmax_p50": pr[0], "rowmax_p99": pr[1], "rowmax_p999": pr[2],
            "frac_rows_gt_1e-2": float((rowmax > 1e-2).float().mean()),
            "frac_rows_gt_3e-2": float((rowmax > 3e-2).float().mean()),
            "frac_rows_gt_1e-1": float((rowmax > 1e-1).float().mean()),
            "rel_l2": float((a - b).norm() / b.norm())}


def sel_agreement(a, b, t0=MTN):
    """Set agreement of two selections on rows t >= t0: identical-row fraction, sum |A n B| / sum |A|, and mean / max
    nodes per row in A but not in B."""
    a = a[:, t0:].reshape(-1, a.shape[-1]).to(torch.int32)
    b = b[:, t0:].reshape(-1, b.shape[-1]).to(torch.int32)
    ar = torch.arange(a.shape[-1], device=a.device, dtype=torch.int32)
    ident = inter_sum = na_sum = 0
    dmax = 0
    for i in range(0, a.shape[0], 1 << 16):
        x, y = a[i:i + (1 << 16)], b[i:i + (1 << 16)]
        ident += int((x == y).all(-1).sum())
        nx = (x >= 0).sum(-1)
        z = torch.cat([torch.where(x >= 0, x, -1 - ar), torch.where(y >= 0, y, -1 - ar.numel() - ar)], -1)
        z = z.sort(-1).values
        inter = (z[:, 1:] == z[:, :-1]).sum(-1)
        inter_sum += int(inter.sum())
        na_sum += int(nx.sum())
        dmax = max(dmax, int((nx - inter).max()))
    rows = a.shape[0]
    return {"rows": rows, "rows_identical": ident / rows, "node_agree": inter_sum / max(na_sum, 1),
            "mean_nodes_differing": (na_sum - inter_sum) / rows, "max_nodes_differing": dmax}


def needle_hits(sel, needle, level, t0=MTN):
    """Fraction of the planted needle nodes (the level-`level` nodes covering the causal part of a row's two needle
    segments) present in the row's selection, rows t >= t0."""
    T = sel.shape[1]
    sh = 4 * level
    t = torch.arange(T, device=sel.device).view(1, T, 1, 1)
    lo = (needle * SEG) >> sh
    hi = torch.minimum(needle * SEG + SEG - 1, t) >> sh
    n = max(1, SEG >> sh)
    nodes = lo.unsqueeze(-1) + torch.arange(n, device=sel.device)
    valid = nodes <= hi.unsqueeze(-1)
    srt = torch.where(sel >= 0, sel.to(torch.int32), torch.iinfo(torch.int32).max)[:, t0:].reshape(-1, sel.shape[-1])
    vals = nodes[:, t0:].reshape(srt.shape[0], -1).to(torch.int32).contiguous()
    valid = valid[:, t0:].reshape(srt.shape[0], -1)
    idx = torch.searchsorted(srt.contiguous(), vals).clamp_max(srt.shape[-1] - 1)
    found = (srt.gather(1, idx) == vals) & valid
    return float(found.sum()) / float(valid.sum())


def sel_invariants(sel, T):
    """Per-level violation counts of the selection invariants: ascending, distinct, -1 tail; rightmost node present;
    min(TK, #candidates) nodes kept; every kept node a candidate."""
    nl = len(_level_sizes(T))
    t = torch.arange(T, device="cuda", dtype=torch.int32).view(1, T, 1, 1)
    big = torch.iinfo(torch.int32).max
    res = {}
    for lvl in range(1, nl):
        x = sel[lvl].to(torch.int32)
        valid = x >= 0
        r = t >> (4 * lvl)                                                        # rightmost node, [1, T, 1, 1]
        if lvl == nl - 1:
            npar = (r >> 4) + 1                                                   # identity-prefix frontier
        else:
            npar = (sel[lvl + 1] >= 0).sum(-1, keepdim=True)
        ncand = (npar - 1) * CR + (r % CR) + 1
        v = {"unsorted_or_dup": int((valid[..., 1:] & ~(x[..., 1:] > x[..., :-1])).sum()),
             "valid_after_tail": int((valid[..., 1:] & ~valid[..., :-1]).sum()),
             "rightmost_missing": int((~(x == r).any(-1)).sum()),
             "count_mismatch": int((valid.sum(-1, keepdim=True) != torch.clamp(ncand, max=TK)).sum()),
             "beyond_rightmost": int((valid & (x > r)).sum())}
        if lvl < nl - 1:
            par = torch.where(sel[lvl + 1] >= 0, sel[lvl + 1].to(torch.int32), big).reshape(-1, TK).contiguous()
            want = torch.where(valid, x >> 4, -1).reshape(-1, TK).contiguous()
            idx = torch.searchsorted(par, want).clamp_max(TK - 1)
            v["parent_not_selected"] = int((valid.reshape(-1, TK) & (par.gather(1, idx) != want)).sum())
        res[str(lvl)] = v
    return res


def collapse_rel(out, v, t0=MTN):
    """|| out - uniform causal mean of v || / || out || on rows t >= t0 (0 = collapsed to mean pooling)."""
    T = v.shape[1]
    cm = v.float().cumsum(1) / torch.arange(1, T + 1, device=v.device, dtype=torch.float32).view(1, T, 1, 1)
    cm = cm.repeat_interleave(G, dim=2)[:, t0:]
    o = out[:, t0:].float()
    return float((o - cm).norm() / o.norm())


def dist(a, b, t0=MTN):
    """|| a - b || over rows t >= t0."""
    return float((a[:, t0:].float() - b[:, t0:].float()).norm())


def oracle_rows(T, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    late = torch.randint(MTN, T, (N_ORACLE_ROWS - 4,), generator=g).tolist()
    return sorted({MTN, MTN + 1, T // 2, T - 1, *late})


def oracle_out(q, k, v, run, rows, no_summaries):
    """The fp32 fixed-selection oracle (recsa.reference.oracles) on `run`'s own selections."""
    T = q.shape[1]
    nl = len(_level_sizes(T))
    plp = {nl - 1: init_prev_selected_parents(T, 1, HKV, TK, CR, nl, q.device)}
    for lvl in range(1, nl):
        plp[lvl - 1] = run.sel[lvl].to(torch.int32)
    params = RecSAParams(compression_rate=CR, max_top_nodes=MTN, top_k_per_layer=TK, rope_base=ROPE,
                         rope_mode="internal", scale=D ** -0.5)
    return fixed_selection_forward_from_parents(q, k, v, plp, params, query_indices=torch.tensor(rows, device="cuda"),
                                                no_summaries=no_summaries)


def oracle_err(q, k, v, run, rows, no_summaries):
    ref = oracle_out(q, k, v, run, rows, no_summaries).float()
    got = run.out[:, rows].float()
    d = (got - ref).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "rel_l2": float((got - ref).norm() / ref.norm())}


class _Bundle:
    """Inputs of one (case, seed) and the runs computed on them so far (each configuration runs once)."""

    def __init__(self, case, seed):
        self.case, self.seed, self.T = case, seed, CASES[case]
        self.q, self.k, self.v, self.needle = make_inputs(self.T, seed)
        self.nl = len(_level_sizes(self.T))
        self.rows = oracle_rows(self.T, seed)
        self._runs = {}
        self._q_ulp = None

    def q_ulp(self):
        """q with every nonzero element moved by one bf16 ulp in a random direction."""
        if self._q_ulp is None:
            g = torch.Generator(device="cuda").manual_seed(4242 + self.seed)
            step = torch.randint(0, 2, self.q.shape, generator=g, device="cuda", dtype=torch.int16) * 2 - 1
            bits = self.q.view(torch.int16)
            self._q_ulp = torch.where(self.q != 0, bits + step, bits).view(torch.bfloat16)
        return self._q_ulp

    def run(self, cfg, no_summaries=False, q_ulp=False):
        key = (cfg, no_summaries, q_ulp)
        if key not in self._runs:
            q = self.q_ulp() if q_ulp else self.q
            self._runs[key] = run_config(cfg, q, self.k, self.v, no_summaries=no_summaries)
        return self._runs[key]


_BUNDLES = {}


def bundle(case, seed=TEST_SEED):
    """The (case, seed) bundle; at most two are cached."""
    if (case, seed) not in _BUNDLES:
        while len(_BUNDLES) >= 2:
            _BUNDLES.pop(next(iter(_BUNDLES)))
        torch.cuda.empty_cache()
        _BUNDLES[(case, seed)] = _Bundle(case, seed)
    return _BUNDLES[(case, seed)]


@pytest.fixture(scope="module", autouse=True)
def _free_bundles():
    yield
    _BUNDLES.clear()
    torch.cuda.empty_cache()


def metrics_a(b):
    dep, fast, dep_ns = b.run("deployed"), b.run("fast"), b.run("deployed", True)
    m = {"out": out_metrics(fast.out, dep.out),
         "sel": {str(l): sel_agreement(fast.sel[l], dep.sel[l]) for l in range(1, b.nl)},
         "needle_hits_deployed": {str(l): needle_hits(dep.sel[l], b.needle, l) for l in range(1, b.nl)},
         "needle_hits_fast": {str(l): needle_hits(fast.sel[l], b.needle, l) for l in range(1, b.nl)},
         "collapse_rel_deployed": collapse_rel(dep.out, b.v),
         "collapse_rel_fast": collapse_rel(fast.out, b.v),
         "oracle_deployed": oracle_err(b.q, b.k, b.v, dep, b.rows, False),
         "oracle_fast": oracle_err(b.q, b.k, b.v, fast, b.rows, False),
         "invariants_fast": sel_invariants(fast.sel, b.T), "invariants_deployed": sel_invariants(dep.sel, b.T),
         "routes_fast": fast.routes, "routes_deployed": dep.routes,
         "wall_s": {"deployed": dep.wall_s, "fast": fast.wall_s}}
    d_fd = max(dist(fast.out, dep.out), 1e-30)
    m["oracle_rel_ratio"] = m["oracle_fast"]["rel_l2"] / max(m["oracle_deployed"]["rel_l2"], 1e-30)
    m["sep_ratio_summaries_vs_fast"] = dist(dep_ns.out, dep.out) / d_fd
    dep_u = b.run("deployed", q_ulp=True)
    m["ulp_ratio"] = dist(dep_u.out, dep.out) / d_fd
    m["sel_ulp"] = {str(l): sel_agreement(dep_u.sel[l], dep.sel[l]) for l in range(1, b.nl)}
    sizes = _level_sizes(b.T)
    m["sel_ulp_ratio"] = {str(l): (1 - m["sel_ulp"][str(l)]["node_agree"]) / max(1 - m["sel"][str(l)]["node_agree"],
                                                                                  1e-9)
                          for l in range(1, b.nl) if sizes[l] > 2 * TK}        # the selective cuts
    return m


def metrics_b(b):
    dep_ns, fast_ns, fast = b.run("deployed", True), b.run("fast", True), b.run("fast")
    m = {"out": out_metrics(fast_ns.out, dep_ns.out),
         "sel": {str(l): sel_agreement(fast_ns.sel[l], dep_ns.sel[l]) for l in range(1, b.nl)},
         "oracle_deployed_nosum": oracle_err(b.q, b.k, b.v, dep_ns, b.rows, True),
         "invariants_fast_nosum": sel_invariants(fast_ns.sel, b.T),
         "routes_fast_nosum": fast_ns.routes, "wall_s": {"deployed_nosum": dep_ns.wall_s, "fast_nosum": fast_ns.wall_s}}
    m["flag_ratio"] = dist(fast_ns.out, fast.out) / max(dist(fast_ns.out, dep_ns.out), 1e-30)
    m["ulp_ratio"] = (dist(b.run("deployed", True, q_ulp=True).out, dep_ns.out)
                      / max(dist(fast_ns.out, dep_ns.out), 1e-30))
    ref_ns = oracle_out(b.q, b.k, b.v, fast_ns, b.rows, True).float()
    ref_soft = oracle_out(b.q, b.k, b.v, fast_ns, b.rows, False).float()
    got = fast_ns.out[:, b.rows].float()
    d = (got - ref_ns).abs()
    m["oracle_fast_nosum"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                              "rel_l2": float((got - ref_ns).norm() / ref_ns.norm())}
    m["oracle_flag_ratio"] = float((got - ref_soft).norm()) / max(float((got - ref_ns).norm()), 1e-30)
    m["oracle_rel_ratio"] = m["oracle_fast_nosum"]["rel_l2"] / max(m["oracle_deployed_nosum"]["rel_l2"], 1e-30)
    return m


def metrics_c_fallback(b, no_summaries):
    tri, bf = b.run("fast_triton", no_summaries), b.run("fast_buildfail", no_summaries)
    same_out = torch.equal(tri.out, bf.out)
    same_sel = all(torch.equal(tri.sel[l], bf.sel[l]) for l in range(1, b.nl))
    return {"buildfail_eq_forced_out": same_out, "buildfail_eq_forced_sel": same_sel,
            "buildfail_flags_after": bf.flags_after, "routes_buildfail": bf.routes, "routes_forced": tri.routes,
            "out_vs_deployed": out_metrics(tri.out, b.run("deployed", no_summaries).out)}


def metrics_c_cuda(b, no_summaries):
    fast, tri = b.run("fast", no_summaries), b.run("fast_triton", no_summaries)
    return {"out": out_metrics(fast.out, tri.out),
            "sel_bitwise": {str(l): bool(torch.equal(fast.sel[l], tri.sel[l])) for l in range(1, b.nl)},
            "oracle_triton": oracle_err(b.q, b.k, b.v, tri, b.rows, no_summaries),
            "routes_fast": fast.routes, "routes_triton": tri.routes,
            "wall_s": {"fast": fast.wall_s, "fast_triton": tri.wall_s}}


def metrics_e(b, cfg):
    dep, run = b.run("deployed"), b.run(cfg)
    return {"out": out_metrics(run.out, dep.out),
            "sel": {str(l): sel_agreement(run.sel[l], dep.sel[l]) for l in range(1, b.nl)},
            "sel_bitwise": {str(l): bool(torch.equal(run.sel[l], dep.sel[l])) for l in range(1, b.nl)},
            "oracle": oracle_err(b.q, b.k, b.v, run, b.rows, False),
            "routes": run.routes, "wall_s": run.wall_s}


THRESH = {
    "a.nl2.L1.node_agree_min": 0.9996,  # 0.9998
    "a.nl2.L1.rows_identical_min": 0.9016,  # 0.9508
    "a.nl2.L1.sel_ulp_ratio_min": 4.5,  # 9.053
    "a.nl2.collapse_rel_min": 0.48,  # 0.9729
    "a.nl2.frac_rows_gt_1e-2": 0.0073,  # 0.003645
    "a.nl2.needle_hits_min": 1.0,  # 1
    "a.nl2.oracle_max_abs": 0.0079,  # 0.003906
    "a.nl2.oracle_rel_l2": 0.0018,  # 0.0008727
    "a.nl2.oracle_rel_ratio": 0.67,  # 0.331
    "a.nl2.p99_elem": 0.0049,  # 0.002441
    "a.nl2.rel_l2": 0.0072,  # 0.003597
    "a.nl2.rowmax_p99": 0.016,  # 0.007812
    "a.nl2.sep_ratio_min": 2.6,  # 5.206
    "a.nl2.ulp_ratio_min": 1.3,  # 2.626
    "a.nl3.L1.node_agree_min": 0.9995,  # 0.9998
    "a.nl3.L1.rows_identical_min": 0.8732,  # 0.9366
    "a.nl3.L1.sel_ulp_ratio_min": 4.3,  # 8.742
    "a.nl3.L2.node_agree_min": 0.9999,  # 1
    "a.nl3.L2.rows_identical_min": 0.9999,  # 1
    "a.nl3.collapse_rel_min": 0.48,  # 0.9772
    "a.nl3.frac_rows_gt_1e-2": 0.0068,  # 0.003388
    "a.nl3.needle_hits_min": 0.9999,  # 1
    "a.nl3.oracle_max_abs": 0.016,  # 0.007812
    "a.nl3.oracle_rel_l2": 0.002,  # 0.0009523
    "a.nl3.oracle_rel_ratio": 0.7,  # 0.3479
    "a.nl3.p99_elem": 0.0054,  # 0.002686
    "a.nl3.rel_l2": 0.0086,  # 0.004293
    "a.nl3.rowmax_p99": 0.016,  # 0.007812
    "a.nl3.sep_ratio_min": 4.6,  # 9.242
    "a.nl3.ulp_ratio_min": 1.3,  # 2.674
    "b.nl2.L1.node_agree_min": 0.9996,  # 0.9998
    "b.nl2.L1.rows_identical_min": 0.9016,  # 0.9508
    "b.nl2.flag_ratio_min": 2.5,  # 5.091
    "b.nl2.frac_rows_gt_1e-2": 0.0078,  # 0.003894
    "b.nl2.oracle_flag_ratio_min": 10.0,  # 20.65
    "b.nl2.oracle_max_abs": 0.016,  # 0.007812
    "b.nl2.oracle_rel_l2": 0.0018,  # 0.0008791
    "b.nl2.oracle_rel_ratio": 0.7,  # 0.3463
    "b.nl2.p99_elem": 0.0049,  # 0.002441
    "b.nl2.rel_l2": 0.0073,  # 0.003632
    "b.nl2.rowmax_p99": 0.016,  # 0.007812
    "b.nl2.ulp_ratio_min": 1.3,  # 2.642
    "b.nl3.L1.node_agree_min": 0.9999,  # 1
    "b.nl3.L1.rows_identical_min": 0.9999,  # 1
    "b.nl3.L2.node_agree_min": 0.9999,  # 1
    "b.nl3.L2.rows_identical_min": 0.9999,  # 1
    "b.nl3.flag_ratio_min": 8.1,  # 16.33
    "b.nl3.frac_rows_gt_1e-2": 0.0001,  # 1.041e-05
    "b.nl3.oracle_flag_ratio_min": 19.0,  # 38.54
    "b.nl3.oracle_max_abs": 0.016,  # 0.007812
    "b.nl3.oracle_rel_l2": 0.0019,  # 0.0009205
    "b.nl3.oracle_rel_ratio": 0.71,  # 0.353
    "b.nl3.p99_elem": 0.004,  # 0.001953
    "b.nl3.rel_l2": 0.0051,  # 0.002519
    "b.nl3.rowmax_p99": 0.0079,  # 0.003906
    "b.nl3.ulp_ratio_min": 2.4,  # 4.998
    "c.nl2.a2.oracle_max_abs": 0.016,  # 0.007812
    "c.nl2.a2.p99_elem": 0.001953125,  # 0
    "c.nl2.a2.rel_l2": 6.6e-05,  # 3.274e-05
    "c.nl2.a2.rowmax_p99": 0.001953125,  # 0
    "c.nl2.softmax.oracle_max_abs": 0.0079,  # 0.003906
    "c.nl2.softmax.p99_elem": 0.001953125,  # 0
    "c.nl2.softmax.rel_l2": 6.5e-05,  # 3.25e-05
    "c.nl2.softmax.rowmax_p99": 0.001953125,  # 0
    "c.nl3.a2.oracle_max_abs": 0.016,  # 0.007812
    "c.nl3.a2.p99_elem": 0.001953125,  # 0
    "c.nl3.a2.rel_l2": 6.7e-05,  # 3.331e-05
    "c.nl3.a2.rowmax_p99": 0.001953125,  # 0
    "c.nl3.softmax.oracle_max_abs": 0.016,  # 0.007812
    "c.nl3.softmax.p99_elem": 0.001953125,  # 0
    "c.nl3.softmax.rel_l2": 6.7e-05,  # 3.31e-05
    "c.nl3.softmax.rowmax_p99": 0.001953125,  # 0
    "cf.nl2.a2.frac_rows_gt_1e-2": 0.0078,  # 0.003893
    "cf.nl2.a2.p99_elem": 0.0049,  # 0.002441
    "cf.nl2.a2.rel_l2": 0.0073,  # 0.003632
    "cf.nl2.softmax.frac_rows_gt_1e-2": 0.0073,  # 0.003645
    "cf.nl2.softmax.p99_elem": 0.0049,  # 0.002441
    "cf.nl2.softmax.rel_l2": 0.0072,  # 0.003597
    "cf.nl3.a2.frac_rows_gt_1e-2": 0.0001,  # 1.041e-05
    "cf.nl3.a2.p99_elem": 0.004,  # 0.001953
    "cf.nl3.a2.rel_l2": 0.0051,  # 0.002519
    "cf.nl3.softmax.frac_rows_gt_1e-2": 0.0068,  # 0.003388
    "cf.nl3.softmax.p99_elem": 0.0054,  # 0.002686
    "cf.nl3.softmax.rel_l2": 0.0086,  # 0.004293
    "e.nl2.acc_fp16.L1.node_agree_min": 1.0,  # 1
    "e.nl2.acc_fp16.frac_rows_gt_1e-2": 0.0001,  # 1.049e-05
    "e.nl2.acc_fp16.oracle_max_abs": 0.0079,  # 0.003906
    "e.nl2.acc_fp16.p99_elem": 0.004,  # 0.001953
    "e.nl2.acc_fp16.rel_l2": 0.005,  # 0.002481
    "e.nl2.fast_noselect.L1.node_agree_min": 1.0,  # 1
    "e.nl2.fast_noselect.frac_rows_gt_1e-2": 0.0001,  # 1.049e-05
    "e.nl2.fast_noselect.oracle_max_abs": 0.0079,  # 0.003906
    "e.nl2.fast_noselect.p99_elem": 0.004,  # 0.001953
    "e.nl2.fast_noselect.rel_l2": 0.005,  # 0.002478
    "e.nl3.acc_fp16.L1.node_agree_min": 1.0,  # 1
    "e.nl3.acc_fp16.L2.node_agree_min": 1.0,  # 1
    "e.nl3.acc_fp16.frac_rows_gt_1e-2": 0.0001,  # 8.99e-06
    "e.nl3.acc_fp16.oracle_max_abs": 0.016,  # 0.007812
    "e.nl3.acc_fp16.p99_elem": 0.004,  # 0.001953
    "e.nl3.acc_fp16.rel_l2": 0.0052,  # 0.002562
    "e.nl3.fast_noselect.L1.node_agree_min": 1.0,  # 1
    "e.nl3.fast_noselect.L2.node_agree_min": 1.0,  # 1
    "e.nl3.fast_noselect.frac_rows_gt_1e-2": 0.0001,  # 8.043e-06
    "e.nl3.fast_noselect.oracle_max_abs": 0.0079,  # 0.003906
    "e.nl3.fast_noselect.p99_elem": 0.004,  # 0.001953
    "e.nl3.fast_noselect.rel_l2": 0.0052,  # 0.002559
}


def _t(key):
    return THRESH[key]


@pytest.mark.parametrize("case", list(CASES))
def test_fast_vs_deployed_softmax(case, record_m1_result):
    b = bundle(case)
    m = metrics_a(b)
    record_m1_result({"check": "prefill_fast_vs_deployed", "case": case, "T": b.T, **m})
    rf, rd = m["routes_fast"], m["routes_deployed"]
    assert rf.get("fast_top_select") == 1 and rf.get("leaf_union") == 1, rf
    assert "recsa_top_select_kernel" not in rf and "recsa_bottom_accumulate_gqa_kernel" not in rf, rf
    if b.nl == 3:
        assert rf.get("fast_mid_select_union", 0) >= 1 and rf.get("mid_acc_union", 0) >= 1, rf
        assert "recsa_mid_select_kernel" not in rf and "recsa_mid_accumulate_kernel" not in rf, rf
    assert rd.get("recsa_top_select_kernel") == 1 and rd.get("recsa_bottom_accumulate_gqa_kernel") == 1, rd
    assert not any(k.startswith(("fast_", "leaf_", "mid_acc_", "cuda:")) for k in rd), rd
    for lvl, hit in {**m["needle_hits_deployed"], **{f"fast{l}": h for l, h in m["needle_hits_fast"].items()}}.items():
        assert hit >= _t(f"a.{case}.needle_hits_min"), (lvl, hit)
    assert min(m["collapse_rel_deployed"], m["collapse_rel_fast"]) >= _t(f"a.{case}.collapse_rel_min"), m
    assert m["sep_ratio_summaries_vs_fast"] >= _t(f"a.{case}.sep_ratio_min"), m["sep_ratio_summaries_vs_fast"]
    assert m["ulp_ratio"] >= _t(f"a.{case}.ulp_ratio_min"), m["ulp_ratio"]
    for lvl, r in m["sel_ulp_ratio"].items():
        assert r >= _t(f"a.{case}.L{lvl}.sel_ulp_ratio_min"), (lvl, r, m["sel"], m["sel_ulp"])
    for who in ("invariants_fast", "invariants_deployed"):
        assert all(n == 0 for lv in m[who].values() for n in lv.values()), (who, m[who])
    # parity (robust to near-tie selection flips)
    o = m["out"]
    assert o["p99_elem"] <= _t(f"a.{case}.p99_elem"), o
    assert o["rowmax_p99"] <= _t(f"a.{case}.rowmax_p99"), o
    assert o["frac_rows_gt_1e-2"] <= _t(f"a.{case}.frac_rows_gt_1e-2"), o
    assert o["rel_l2"] <= _t(f"a.{case}.rel_l2"), o
    for lvl, s in m["sel"].items():
        assert s["node_agree"] >= _t(f"a.{case}.L{lvl}.node_agree_min"), (lvl, s)
        assert s["rows_identical"] >= _t(f"a.{case}.L{lvl}.rows_identical_min"), (lvl, s)
    assert m["oracle_fast"]["max_abs"] <= _t(f"a.{case}.oracle_max_abs"), m["oracle_fast"]
    assert m["oracle_fast"]["rel_l2"] <= _t(f"a.{case}.oracle_rel_l2"), m["oracle_fast"]
    # ... and no less accurate than the deployed TF32 kernels on theirs
    assert m["oracle_rel_ratio"] <= _t(f"a.{case}.oracle_rel_ratio"), (m["oracle_fast"], m["oracle_deployed"])


@pytest.mark.parametrize("case", list(CASES))
def test_fast_a2_no_summaries(case, record_m1_result):
    b = bundle(case)
    m = metrics_b(b)
    record_m1_result({"check": "prefill_fast_a2_vs_deployed", "case": case, "T": b.T, **m})
    r = m["routes_fast_nosum"]
    assert r.get("fast_top_select") == 1 and r.get("leaf_union") == 1, r
    assert m["flag_ratio"] >= _t(f"b.{case}.flag_ratio_min"), m
    assert m["oracle_flag_ratio"] >= _t(f"b.{case}.oracle_flag_ratio_min"), m
    assert m["ulp_ratio"] >= _t(f"b.{case}.ulp_ratio_min"), m["ulp_ratio"]
    assert all(n == 0 for lv in m["invariants_fast_nosum"].values() for n in lv.values()), m["invariants_fast_nosum"]
    o = m["out"]
    assert o["p99_elem"] <= _t(f"b.{case}.p99_elem"), o
    assert o["rowmax_p99"] <= _t(f"b.{case}.rowmax_p99"), o
    assert o["frac_rows_gt_1e-2"] <= _t(f"b.{case}.frac_rows_gt_1e-2"), o
    assert o["rel_l2"] <= _t(f"b.{case}.rel_l2"), o
    for lvl, s in m["sel"].items():
        assert s["node_agree"] >= _t(f"b.{case}.L{lvl}.node_agree_min"), (lvl, s)
    assert m["oracle_fast_nosum"]["max_abs"] <= _t(f"b.{case}.oracle_max_abs"), m["oracle_fast_nosum"]
    assert m["oracle_fast_nosum"]["rel_l2"] <= _t(f"b.{case}.oracle_rel_l2"), m["oracle_fast_nosum"]
    assert m["oracle_rel_ratio"] <= _t(f"b.{case}.oracle_rel_ratio"), (m["oracle_fast_nosum"],
                                                                        m["oracle_deployed_nosum"])


_C_PARAMS = [pytest.param(c, ns, id=f"{c}-{'a2' if ns else 'softmax'}") for c in CASES for ns in (False, True)]


@pytest.mark.parametrize("case,no_summaries", _C_PARAMS)
def test_cuda_build_failure_falls_back_to_triton(case, no_summaries, capsys):
    before = (MA.LEAF_CUDA, MA._LEAF_CUDA_BROKEN, FS.FAST_TOP_B_CUDA, FS.FAST_MIDA_CUDA, FS.FAST_TOPA_CUDA, FS._TB,
              LC.build, TS.build, P.TOPACC_CUDA)
    b = bundle(case)
    m = metrics_c_fallback(b, no_summaries)
    assert "using the Triton" in capsys.readouterr().out, "the fallback did not report itself"
    assert m["buildfail_eq_forced_out"] and m["buildfail_eq_forced_sel"], m
    fl = m["buildfail_flags_after"]
    assert fl["_LEAF_CUDA_BROKEN"] and fl["FAST_TOP_B_CUDA"] == 0 and fl["FAST_TOPA_CUDA"] == 0, fl
    for routes in (m["routes_buildfail"], m["routes_forced"]):
        assert not any(k.startswith("cuda:") for k in routes), routes
        assert routes.get("fast_top_select") == 1 and routes.get("leaf_union") == 1, routes
    after = (MA.LEAF_CUDA, MA._LEAF_CUDA_BROKEN, FS.FAST_TOP_B_CUDA, FS.FAST_MIDA_CUDA, FS.FAST_TOPA_CUDA, FS._TB,
             LC.build, TS.build, P.TOPACC_CUDA)
    assert after == before, (before, after)
    key = f"cf.{case}.{'a2' if no_summaries else 'softmax'}"
    o = m["out_vs_deployed"]
    assert o["p99_elem"] <= _t(f"{key}.p99_elem"), o
    assert o["frac_rows_gt_1e-2"] <= _t(f"{key}.frac_rows_gt_1e-2"), o
    assert o["rel_l2"] <= _t(f"{key}.rel_l2"), o


@pytest.mark.parametrize("case,no_summaries", _C_PARAMS)
def test_cuda_kernels_vs_triton_fallback(case, no_summaries, record_m1_result):
    why = cuda_side_unavailable()
    if why:
        pytest.skip(why)
    b = bundle(case)
    m = metrics_c_cuda(b, no_summaries)
    record_m1_result({"check": "prefill_fast_cuda_vs_triton", "case": case, "no_summaries": no_summaries, **m})
    rf = m["routes_fast"]
    assert rf.get("cuda:leaf_union_cuda") == 1 and rf.get("cuda:top_finalize", 0) >= 1, rf
    assert rf.get("cuda:top_score", 0) >= 1, rf
    if not no_summaries:
        assert rf.get("cuda:top_accum_cuda") == 1, rf
        if b.nl == 3:
            assert rf.get("cuda:mid_score", 0) >= 1 and rf.get("cuda:mid_finalize", 0) >= 1, rf
    assert not any(k.startswith("cuda:") for k in m["routes_triton"]), m["routes_triton"]
    assert all(m["sel_bitwise"].values()), m["sel_bitwise"]
    key = f"c.{case}.{'a2' if no_summaries else 'softmax'}"
    o = m["out"]
    assert o["p99_elem"] <= _t(f"{key}.p99_elem"), o
    assert o["rowmax_p99"] <= _t(f"{key}.rowmax_p99"), o
    assert o["rel_l2"] <= _t(f"{key}.rel_l2"), o
    assert m["oracle_triton"]["max_abs"] <= _t(f"{key}.oracle_max_abs"), m["oracle_triton"]


def _sel_lists(T, n_parents, seed, full_every=3):
    """Random int32 [1, T, 8, 256] selection lists: ascending distinct parents < n_parents with a -1 tail, overlapping
    between neighbouring queries."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    nb = -(-T // 16)
    pref = torch.rand(nb, HKV, n_parents, generator=g, device="cuda").repeat_interleave(16, 0)[:T]
    s = pref + 0.3 * torch.rand(T, HKV, n_parents, generator=g, device="cuda")
    kk = min(TK, n_parents)
    idx = s.topk(kk, dim=-1).indices.to(torch.int32)
    cnt = torch.randint(1, kk + 1, (T, HKV, 1), generator=g, device="cuda")
    cnt[::full_every] = kk
    big = torch.iinfo(torch.int32).max
    vals = torch.where(torch.arange(kk, device="cuda") < cnt, idx, big).sort(-1).values
    sel = torch.full((T, HKV, TK), -1, dtype=torch.int32, device="cuda")
    sel[..., :kk] = torch.where(vals == big, -1, vals)
    return sel.unsqueeze(0).contiguous()


def _children_lists(sel, seed):
    """The next level's emitted lists: per row <= 256 children (parent * 16 + c) of the row's own parents."""
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    cand = sel.unsqueeze(-1) * CR + torch.arange(CR, device="cuda", dtype=torch.int32)        # [1, T, H, TK, CR]
    ok = (sel >= 0).unsqueeze(-1).expand_as(cand)
    cand, ok = cand.flatten(-2), ok.flatten(-2)
    score = torch.where(ok, torch.rand(cand.shape, generator=g, device="cuda"), -1.0)
    top = score.topk(TK, dim=-1)
    big = torch.iinfo(torch.int32).max
    pick = torch.where(top.values >= 0, cand.gather(-1, top.indices), big).sort(-1).values
    return torch.where(pick == big, -1, pick).contiguous()


_UNION_CASES = [pytest.param(T, n, id=f"T{T}-P{n}") for T, n in ((1000, 258), (1000, 2048), (777, 4128))]


@pytest.mark.parametrize("T,n_parents", _UNION_CASES)
def test_union_lists_builders_bitwise(T, n_parents):
    """merge_acc.union_lists: the Triton builder matches the torch reference, with and without umax."""
    sel = _sel_lists(T, n_parents, seed=T + n_parents)
    for umax in (None, n_parents):
        ref = MA.union_lists_torch(sel, 16, umax)
        tri = MA.union_lists_triton(sel, 16, umax)
        assert all(torch.equal(x, y) for x, y in zip(ref, tri)), f"Triton != torch (umax={umax})"
        assert int(ref[2].min()) >= 1 and int(ref[2].max()) <= n_parents
        assert int(ref[2].max()) > 256, "vacuous: no block's union exceeds one row"


@pytest.mark.parametrize("T,n_parents", _UNION_CASES)
def test_union_lists_cuda_bitwise(T, n_parents):
    """The CUDA builders in recsa/cuda/union_build.cu match the torch reference; their query positions match
    merge_acc.query_positions."""
    why = cuda_side_unavailable()
    if why:
        pytest.skip(why)
    sel = _sel_lists(T, n_parents, seed=T + n_parents)
    for umax in (None, n_parents):
        ref = MA.union_lists_torch(sel, 16, umax)
        for np_ in (None, n_parents):
            cu = LC.union_lists16(sel, pad=True, umax=umax, n_parents=np_)
            assert all(torch.equal(x, y) for x, y in zip(ref, cu)), f"CUDA != torch (umax={umax}, n_parents={np_})"
    *_, posq = LC.union_lists16(sel, pad=False, posq_t_base=T + 5, n_parents=n_parents)
    assert torch.equal(posq, MA.query_positions(sel, T + 5, CR, 1))


@pytest.mark.parametrize("T,n_parents", _UNION_CASES[:2])
def test_union_xmask_builders_bitwise(T, n_parents):
    """merge_acc.union_xmask: Triton matches torch, and so does the CUDA builder when available."""
    sel = _sel_lists(T, n_parents, seed=7 * T + n_parents)
    nxt = _children_lists(sel, seed=T)
    ul, um, uc = MA.union_lists_torch(sel, 16, n_parents)
    ref = MA.union_xmask_torch(nxt, ul, 16, CR)
    tri = MA.union_xmask_triton(nxt, ul, uc, 16, CR)
    assert torch.equal(ref, tri), "Triton != torch"
    assert int(ref.ne(0).sum()) > 0, "vacuous: no exclusion bit set"
    why = cuda_side_unavailable()
    if why:
        pytest.skip(f"Triton == torch verified; CUDA side: {why}")
    cu = LC.union_xmask16(nxt, ul, uc, CR, n_parents)
    assert cu is not None, "union_xmask16 declined a row that fits its shared memory"
    assert torch.equal(ref, cu), "CUDA != torch"


def _counted(fn, name, counter):
    @functools.wraps(fn)
    def run(*args, **kw):
        counter[name] += 1
        return fn(*args, **kw)
    return run


def _probe(seed, mlp):
    g = torch.Generator(device="cuda").manual_seed(seed)
    f32 = dict(device="cuda", dtype=torch.float32)
    w = 0.3 * torch.randn(D, generator=g, **f32)
    tau = torch.tensor(1.7, **f32)
    gate = torch.tensor(0.8, **f32)
    px = None
    if mlp:
        px = {"mlp_w1": 0.05 * torch.randn(64, 2 * D, generator=g, **f32),
              "mlp_b1": 0.05 * torch.randn(64, generator=g, **f32),
              "mlp_w2": 0.05 * torch.randn(D, 64, generator=g, **f32)}
    return w, tau, gate, px


@pytest.mark.parametrize("mode", ["shared_std", "mlp_pooler"])
@pytest.mark.parametrize("gated", [True, False], ids=["gate", "nogate"])
def test_pool_fused_bitwise(mode, gated, monkeypatch):
    """recsa/pool_fused.py vs the torch pooling on levels 1 and 2 of the prefill pyramid; k and v are equal."""
    q, k, v, _ = make_inputs(66049, seed=3)
    del q
    w, tau, gate, px = _probe(11, mode == "mlp_pooler")
    gate = gate if gated else None

    def pool(ck, cv, fused):
        monkeypatch.setattr(PF, "POOL_FUSED", 1 if fused else 0)
        if mode == "shared_std":
            return _pool_one_level(ck, cv, CR, w, tau, standardize=True, probe_gate=gate)
        return _pool_one_level_mlp(ck, cv, CR, "mlp_pooler", w, tau, gate, px)

    calls = collections.Counter()
    for name in ("pool_std_fused", "pool_mlp_fused"):
        monkeypatch.setattr(PF, name, _counted(getattr(PF, name), name, calls))
    monkeypatch.setattr(PF, "POOL_FUSED", 1)
    if not PF.fused_ok(k, v, CR, w, tau, gate, k.shape[1]):
        pytest.skip("recsa/pool_fused.py not eligible on this build (certified torch / triton versions only)")
    fused_name = "pool_std_fused" if mode == "shared_std" else "pool_mlp_fused"
    for n in (66048, 66049):
        ck, cv = k[:, :n].contiguous(), v[:, :n].contiguous()
        for level in (1, 2):
            n0 = calls[fused_name]
            rk, rv = pool(ck, cv, False)
            assert calls[fused_name] == n0, "the reference ran the fused path"
            fk, fv = pool(ck, cv, True)
            assert calls[fused_name] == n0 + 1, "the fused path did not run"
            assert torch.isfinite(rk.float()).all()
            assert torch.equal(rk, fk) and torch.equal(rv, fv), f"n={n} level {level}"
            ck, cv = rk.contiguous(), rv.contiguous()


_E_PARAMS = [pytest.param(c, cfg, id=f"{c}-{cfg}") for c in CASES
             for cfg in ("fast_noselect", "acc_fp16")]


@pytest.mark.parametrize("case,cfg", _E_PARAMS)
def test_kill_switch_configs_vs_deployed(case, cfg, record_m1_result):
    b = bundle(case)
    m = metrics_e(b, cfg)
    record_m1_result({"check": "prefill_kill_switch_vs_deployed", "case": case, "cfg": cfg, **m})
    r = m["routes"]
    if cfg == "fast_noselect":
        assert r.get("leaf_union") == 1 and "fast_top_select" not in r and "fast_mid_select_union" not in r, r
    else:
        assert r.get("recsa_bottom_accumulate_gqa_kernel") == 1 and "leaf_union" not in r, r
    key = f"e.{case}.{cfg}"
    assert all(m["sel_bitwise"].values()), m["sel_bitwise"]
    o = m["out"]
    assert o["p99_elem"] <= _t(f"{key}.p99_elem"), o
    assert o["frac_rows_gt_1e-2"] <= _t(f"{key}.frac_rows_gt_1e-2"), o
    assert o["rel_l2"] <= _t(f"{key}.rel_l2"), o
    assert m["oracle"]["max_abs"] <= _t(f"{key}.oracle_max_abs"), m["oracle"]


def _round_up(x, sig=2):
    """x rounded up to `sig` significant digits."""
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x)) - sig + 1
    return float(f"{math.ceil(x / 10 ** e)}e{e}")


def measure(seeds, cases):
    """Every metric the tests assert, per seed (seed 0 = the tests' seed)."""
    res = {}
    cuda_why = cuda_side_unavailable()
    for case in cases:
        res[case] = {}
        for seed in seeds:
            t0 = time.time()
            b = bundle(case, seed)
            r = {"a": metrics_a(b), "b": metrics_b(b)}
            r["c_fallback"] = {ns: metrics_c_fallback(b, ns == "a2") for ns in ("softmax", "a2")}
            if not cuda_why:
                r["c_cuda"] = {ns: metrics_c_cuda(b, ns == "a2") for ns in ("softmax", "a2")}
            r["e"] = {cfg: metrics_e(b, cfg) for cfg in ("fast_noselect", "acc_fp16")}
            r["bundle_wall_s"] = time.time() - t0
            res[case][str(seed)] = r
            print(f"[measure] {case} seed {seed}: {r['bundle_wall_s']:.1f} s  a.out={r['a']['out']}", flush=True)
    return res, cuda_why


def _derive(res):
    th = {}

    def up(key, vals):
        thr = _round_up(2 * max(vals))
        if ".frac_rows_gt_" in key:
            thr = max(thr, 1e-4)
        if key.endswith(("p99_elem", "rowmax_p99")):
            thr = max(thr, 2.0 ** -9)
        th[key] = {"threshold": thr, "worst": max(vals), "values": list(vals)}

    def frac_floor(key, vals):          # rounded down to 4 decimals
        th[key] = {"threshold": max(0.0, math.floor((1 - 2 * (1 - min(vals))) * 1e4) / 1e4), "worst": min(vals),
                   "values": list(vals)}

    def ratio_floor(key, vals):         # rounded down to 2 significant digits
        x = min(vals) / 2
        e = math.floor(math.log10(x)) - 1 if x > 0 else 0
        th[key] = {"threshold": float(f"{math.floor(x / 10 ** e)}e{e}") if x > 0 else 0.0, "worst": min(vals),
                   "values": list(vals)}

    for case, per_seed in res.items():
        rs = list(per_seed.values())
        a = [r["a"] for r in rs]
        b = [r["b"] for r in rs]
        frac_floor(f"a.{case}.needle_hits_min",
                   [h for x in a for h in (*x["needle_hits_deployed"].values(), *x["needle_hits_fast"].values())])
        ratio_floor(f"a.{case}.collapse_rel_min",
                    [min(x["collapse_rel_deployed"], x["collapse_rel_fast"]) for x in a])
        ratio_floor(f"a.{case}.sep_ratio_min", [x["sep_ratio_summaries_vs_fast"] for x in a])
        ratio_floor(f"a.{case}.ulp_ratio_min", [x["ulp_ratio"] for x in a])
        ratio_floor(f"b.{case}.ulp_ratio_min", [x["ulp_ratio"] for x in b])
        for lvl in a[0]["sel_ulp_ratio"]:
            ratio_floor(f"a.{case}.L{lvl}.sel_ulp_ratio_min", [x["sel_ulp_ratio"][lvl] for x in a])
        for part, xs in (("a", a), ("b", b)):
            for mk in ("p99_elem", "rowmax_p99", "frac_rows_gt_1e-2", "rel_l2"):
                up(f"{part}.{case}.{mk}", [x["out"][mk] for x in xs])
            for lvl in xs[0]["sel"]:
                frac_floor(f"{part}.{case}.L{lvl}.node_agree_min", [x["sel"][lvl]["node_agree"] for x in xs])
                frac_floor(f"{part}.{case}.L{lvl}.rows_identical_min", [x["sel"][lvl]["rows_identical"] for x in xs])
        up(f"a.{case}.oracle_max_abs", [x["oracle_fast"]["max_abs"] for x in a])
        up(f"a.{case}.oracle_rel_l2", [x["oracle_fast"]["rel_l2"] for x in a])
        up(f"a.{case}.oracle_rel_ratio", [x["oracle_rel_ratio"] for x in a])
        up(f"b.{case}.oracle_rel_ratio", [x["oracle_rel_ratio"] for x in b])
        up(f"b.{case}.oracle_max_abs", [x["oracle_fast_nosum"]["max_abs"] for x in b])
        up(f"b.{case}.oracle_rel_l2", [x["oracle_fast_nosum"]["rel_l2"] for x in b])
        ratio_floor(f"b.{case}.flag_ratio_min", [x["flag_ratio"] for x in b])
        ratio_floor(f"b.{case}.oracle_flag_ratio_min", [x["oracle_flag_ratio"] for x in b])
        for ns in ("softmax", "a2"):
            cs = [r["c_fallback"][ns]["out_vs_deployed"] for r in rs]
            for mk in ("p99_elem", "frac_rows_gt_1e-2", "rel_l2"):
                up(f"cf.{case}.{ns}.{mk}", [x[mk] for x in cs])
        if "c_cuda" in rs[0]:
            for ns in ("softmax", "a2"):
                cs = [r["c_cuda"][ns] for r in rs]
                for mk in ("p99_elem", "rowmax_p99", "rel_l2"):
                    up(f"c.{case}.{ns}.{mk}", [x["out"][mk] for x in cs])
                up(f"c.{case}.{ns}.oracle_max_abs", [x["oracle_triton"]["max_abs"] for x in cs])
        for cfg in rs[0]["e"]:
            es = [r["e"][cfg] for r in rs]
            for mk in ("p99_elem", "frac_rows_gt_1e-2", "rel_l2"):
                up(f"e.{case}.{cfg}.{mk}", [x["out"][mk] for x in es])
            up(f"e.{case}.{cfg}.oracle_max_abs", [x["oracle"]["max_abs"] for x in es])
            for lvl in es[0]["sel"]:
                frac_floor(f"e.{case}.{cfg}.L{lvl}.node_agree_min", [x["sel"][lvl]["node_agree"] for x in es])
    return th


def _main():
    ap = argparse.ArgumentParser(description="measure the tolerances of tests/test_prefill_fast.py")
    ap.add_argument("--measure", required=True, help="output JSON")
    ap.add_argument("--seeds", default=",".join(map(str, MEASURE_SEEDS)))
    ap.add_argument("--cases", default=",".join(CASES))
    args = ap.parse_args()
    from recsa.testing.reporting import collect_env
    seeds = [int(s) for s in args.seeds.split(",")]
    cases = args.cases.split(",")
    t0 = time.time()
    res, cuda_why = measure(seeds, cases)
    env = collect_env()
    env["nvcc"] = LC.NVCC
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    env["RECSA_PREFILL_FAST"] = os.environ.get("RECSA_PREFILL_FAST")
    out = {"what": "tests/test_prefill_fast.py tolerance evidence (every asserted metric, per seed)",
           "geometry": {"CR": CR, "TK": TK, "MTN": MTN, "HQ": HQ, "HKV": HKV, "D": D, "rope_base": ROPE,
                        "cases": {c: CASES[c] for c in cases}},
           "inputs": {"SEG": SEG, "LOWF": list(LOWF), "A_K": A_K, "B_Q": B_Q, "oracle_rows": N_ORACLE_ROWS},
           "seeds": seeds, "test_seed": TEST_SEED, "cuda_side_unavailable": cuda_why,
           "rule": "upper = round_up_2sig(2 x max over seeds); fraction floor = 1 - 2 (1 - min); "
                   "ratio floor = min / 2",
           "derived": _derive(res), "per_seed": res, "env": env, "wall_s": time.time() - t0}
    out["derived_thresholds"] = {k: v["threshold"] for k, v in out["derived"].items()}
    Path(args.measure).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["derived_thresholds"], indent=1))


if __name__ == "__main__":
    _main()
