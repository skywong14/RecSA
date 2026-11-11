"""Operator variant without summaries ("summaries": "none"): forward and prefill."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from recsa.reference.oracles import RecSAParams, fixed_selection_forward_from_parents  # noqa: E402
from recsa.testing.reporting import diff_stats  # noqa: E402
from recsa.naive import recsa_forward_naive  # noqa: E402
from recsa.parallel import recsa_forward  # noqa: E402
from recsa.probe_pool import compute_num_layers  # noqa: E402

pytestmark = pytest.mark.gpu

CR, TK, MTN = 16, 256, 4096
T_NL2, T_NL3 = 8192, 66048
D = 128
ROPE = 5e6

PATHS = {
    "train": dict(),
    "ns": dict(mid_ns_wide=True),
    "legacy_top": dict(packed_top=False),
    "eval_split": dict(mid_split=True),    # eval kernels: M-select + M-accum (split)
}


def _inputs(T, hq, hkv, dtype, seed, v_scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, T, hq, D, generator=g, device="cuda", dtype=torch.float32).to(dtype)
    k = torch.randn(1, T, hkv, D, generator=g, device="cuda", dtype=torch.float32).to(dtype)
    v = (torch.randn(1, T, hkv, D, generator=g, device="cuda", dtype=torch.float32) * v_scale).to(dtype)
    return q, k, v


def _fwd(q, k, v, path, no_summaries):
    with torch.no_grad():
        out, ctx = recsa_forward(
            q, k, v, compression_rate=CR, max_top_nodes=MTN, top_k_per_layer=TK,
            rope_base=ROPE, no_summaries=no_summaries,
            _save_for_backward=True, _save_parents=True, **PATHS[path])
    return out, ctx


def _params():
    return RecSAParams(compression_rate=CR, max_top_nodes=MTN, top_k_per_layer=TK,
                       rope_base=ROPE, scale=D ** -0.5)


def _rows(T):
    """Rows covering t < mtn, the first row with a dropped top candidate, mid-sequence and the tail."""
    rows = {0, 15, MTN - 1, MTN, MTN + 17, T // 2, T // 2 + 5, T - 257, T - 2, T - 1}
    g = torch.Generator().manual_seed(T)
    rows |= set(torch.randint(MTN, T, (6,), generator=g).tolist())
    return sorted(r for r in rows if 0 <= r < T)


def _parents_match_rows(pa: dict, pb: dict, rows, nl):
    """Per row, whether every level's selected parents match between two parent dicts."""
    ok = []
    for r in rows:
        same = all(torch.equal(pa[li][:, r], pb[li][:, r]) for li in range(nl))
        ok.append(same)
    return ok


MIN_SEP = {2: 2.0, 3: 10.0}
MIN_SEP_NAIVE_NL2 = 1.0


def _separation(out_rows, ref_on, ref_off):
    """Relative L2 distances (kernel to flag-on ref, kernel to flag-off ref), normalised by the flag-on ref."""
    o, a, b = out_rows.float(), ref_on.float(), ref_off.float()
    n = a.norm()
    return float((o - a).norm() / n), float((o - b).norm() / n)


# ---------------------------------------------------------------- oracles

ORACLE_CASES = [
    pytest.param(T_NL2, 4, 2, torch.float32, id="nl2-fp32"),
    pytest.param(T_NL2, 8, 2, torch.bfloat16, id="nl2-bf16"),
    pytest.param(T_NL3, 4, 2, torch.float32, id="nl3-fp32"),
    pytest.param(T_NL3, 8, 2, torch.bfloat16, id="nl3-bf16"),
]


@pytest.mark.parametrize("T,hq,hkv,dtype", ORACLE_CASES)
@pytest.mark.parametrize("path", list(PATHS))
def test_kernel_matches_fixed_selection_oracle(T, hq, hkv, dtype, path, record_m1_result):
    q, k, v = _inputs(T, hq, hkv, dtype, seed=1000 + T + hq)
    out, ctx = _fwd(q, k, v, path, no_summaries=True)
    nl = ctx["num_layers"]
    assert nl == compute_num_layers(T, CR, MTN) and nl in (2, 3)
    assert ctx["no_summaries"] is True
    rows_l = _rows(T)
    rows = torch.tensor(rows_l, device="cuda")
    ref = fixed_selection_forward_from_parents(
        q, k, v, ctx["per_layer_parents"], _params(), query_indices=rows, no_summaries=True)
    ref_off = fixed_selection_forward_from_parents(
        q, k, v, ctx["per_layer_parents"], _params(), query_indices=rows, no_summaries=False)
    late = torch.tensor([i for i, r in enumerate(rows_l) if r >= MTN], device="cuda")
    d_on, d_off = _separation(out[:, rows][:, late], ref[:, late], ref_off[:, late])
    st = diff_stats(out[:, rows], ref)
    print(f"[no-summaries oracle] {path} T={T} {dtype}: {st}; separation on {late.numel()} rows t>=mtn: "
          f"rel L2 to flag-on {d_on:.3g}, to flag-off {d_off:.3g} ({d_off / d_on:.3g}x)")
    record_m1_result({"check": "a2_fixed_selection_oracle", "path": path, "T": T,
                      "dtype": str(dtype), "sep_rel_on": d_on, "sep_rel_off": d_off, **st})
    assert torch.isfinite(out).all()
    assert st["p99_abs"] < 1e-2, st
    assert late.numel() > 0 and d_off >= MIN_SEP[nl] * d_on, \
        f"kernel not separated from the flag-off operator: {d_off:.3g} < {MIN_SEP[nl]} x {d_on:.3g}"


NAIVE_CASES = [
    # NL=2 at T = mtn + 256: every row computed
    pytest.param(MTN + 256, 4, 2, torch.float32, None, id="nl2-fp32-allrows"),
    pytest.param(MTN + 256, 4, 2, torch.bfloat16, None, id="nl2-bf16-allrows"),
    # NL=3: the reference's row subset (selection and attention are per row)
    pytest.param(T_NL3, 4, 2, torch.float32, "subset", id="nl3-fp32-rows"),
]

TIE_QUANTUM = 2.0 ** -(23 - 12)
TIE_BOUND = 4 * TIE_QUANTUM


def _naive_level_importance(q, lk, cos, sin, r, h, G, level, frontier):
    """Naive-reference importances {node index: importance} for row r, kv head h at a non-bottom level."""
    from recsa.naive import _apply_rope
    vp = frontier[frontier >= 0].long()
    lp = CR ** level
    nc = (vp.numel() - 1) * CR + (r // lp) % CR + 1
    cp = torch.arange(nc)
    ni = vp[cp // CR] * CR + cp % CR
    kr = _apply_rope(lk[level][0, ni, h].float(), cos[cp], sin[cp])
    rp = nc - 1
    qr = _apply_rope(q[0, r, h * G:(h + 1) * G].float(), cos[rp], sin[rp]) * float(D ** -0.5)
    sc = qr @ kr.T
    imp = torch.exp(sc - sc.max()).sum(dim=0)
    return {int(n): float(x) for n, x in zip(ni.tolist(), imp.tolist())}


@pytest.mark.parametrize("T,hq,hkv,dtype,rowsel", NAIVE_CASES)
def test_kernel_matches_naive_reference(T, hq, hkv, dtype, rowsel, record_m1_result):
    """Compare against the naive reference, which runs its own selection."""
    from recsa.naive import _build_rope_cache, _build_tree
    q, k, v = _inputs(T, hq, hkv, dtype, seed=2000 + T)
    out, ctx = _fwd(q, k, v, "train", no_summaries=True)
    nl = ctx["num_layers"]
    rows = list(range(T)) if rowsel is None else _rows(T)
    qc, kc, vc = q.cpu().float(), k.cpu().float(), v.cpu().float()
    ref, trace = recsa_forward_naive(
        qc, kc, vc, compression_rate=CR, max_top_nodes=MTN, top_k_per_layer=TK,
        scale=D ** -0.5, rope_base=ROPE, return_trace=True, no_summaries=True,
        query_rows=None if rowsel is None else rows)
    ref_off, trace_off = recsa_forward_naive(
        qc, kc, vc, compression_rate=CR, max_top_nodes=MTN, top_k_per_layer=TK,
        scale=D ** -0.5, rope_base=ROPE, return_trace=True, no_summaries=False,
        query_rows=None if rowsel is None else rows)
    naive_par = {nl - 1: ctx["per_layer_parents"][nl - 1].cpu()}
    for li in range(nl - 1, 0, -1):
        naive_par[li - 1] = trace["layers"][li].selected_parents
        assert torch.equal(trace_off["layers"][li].selected_parents, naive_par[li - 1]), \
            "the naive reference's selection depends on the summaries flag"
    kern_par = {li: p.cpu() for li, p in ctx["per_layer_parents"].items()}
    match = _parents_match_rows(kern_par, naive_par, rows, nl)
    good = [r for r, m in zip(rows, match) if m]
    bad = [r for r, m in zip(rows, match) if not m]
    # near-tie classification of every mismatched (row, kv head)
    lk, _ = _build_tree(kc, vc, CR, MTN)
    cos, sin = _build_rope_cache(D, MTN + 1024, torch.device("cpu"), ROPE)
    G = hq // hkv
    worst_gap, n_pairs = 0.0, 0
    for r in bad:
        for h in range(hkv):
            for L in range(nl - 1, 0, -1):
                fr = kern_par[L][0, r, h]
                assert torch.equal(fr, naive_par[L][0, r, h]), (r, h, L)  # first difference is a selection
                sk, sn = kern_par[L - 1][0, r, h], naive_par[L - 1][0, r, h]
                if torch.equal(sk, sn):
                    continue
                imp = _naive_level_importance(qc, lk, cos, sin, r, h, G, L, fr)
                a = {int(x) for x in sk.tolist() if x >= 0}
                b = {int(x) for x in sn.tolist() if x >= 0}
                kern_only, naive_only = a - b, b - a
                assert len(kern_only) == len(naive_only) > 0, (r, h, L, len(kern_only), len(naive_only))
                gap = max(imp[n] for n in naive_only) / min(imp[n] for n in kern_only) - 1.0
                worst_gap = max(worst_gap, gap)
                n_pairs += len(kern_only)
                break
    late_good = [r for r in good if r >= MTN]
    st = diff_stats(out[:, good].cpu().float(), ref[:, good].to(out.dtype).float())
    d_on, d_off = _separation(out[:, late_good].cpu(), ref[:, late_good].to(out.dtype),
                              ref_off[:, late_good].to(out.dtype))
    k_sep = MIN_SEP_NAIVE_NL2 if nl == 2 else MIN_SEP[nl]
    print(f"[no-summaries naive] T={T} {dtype}: {len(good)}/{len(rows)} rows with identical selection "
          f"({len(late_good)} with dropped candidates); {len(bad)} near-tie rows, {n_pairs} swapped "
          f"candidates, worst exact-importance gap {worst_gap:.3g} (bound {TIE_BOUND:.3g}); {st}; "
          f"separation rel L2 to flag-on {d_on:.3g}, to flag-off {d_off:.3g} ({d_off / d_on:.3g}x)")
    record_m1_result({"check": "a2_naive_reference", "T": T, "dtype": str(dtype), "rows": len(rows),
                      "selection_mismatch_rows": len(bad), "swapped_candidates": n_pairs,
                      "worst_tie_gap": worst_gap, "sep_rel_on": d_on, "sep_rel_off": d_off, **st})
    assert late_good, "no row with dropped candidates matched — the gate would be vacuous"
    assert worst_gap <= TIE_BOUND, f"a selection difference is NOT a near-tie: gap {worst_gap:g}"
    assert st["p99_abs"] < 3e-2, st
    assert d_off > k_sep * d_on, \
        f"kernel not separated from the flag-off operator: {d_off:.3g} <= {k_sep} x {d_on:.3g}"


@pytest.mark.parametrize("T,dtype", [
    pytest.param(T_NL2, torch.bfloat16, id="nl2-bf16"),
    pytest.param(T_NL3, torch.float32, id="nl3-fp32"),
])
@pytest.mark.parametrize("path", list(PATHS))
def test_flag_is_identity_when_nothing_is_unselected(T, dtype, path):
    q, k, v = _inputs(T, 8, 2, dtype, seed=3000 + T)
    on, ctx_on = _fwd(q, k, v, path, no_summaries=True)
    off, ctx_off = _fwd(q, k, v, path, no_summaries=False)
    for li in ctx_on["per_layer_parents"]:
        assert torch.equal(ctx_on["per_layer_parents"][li], ctx_off["per_layer_parents"][li]), \
            f"level {li}: the flag changed selection"
    assert torch.equal(on[:, :MTN], off[:, :MTN]), (
        f"rows t < {MTN}: max|d|={(on[:, :MTN].float() - off[:, :MTN].float()).abs().max().item():g}")
    assert torch.equal(ctx_on["global_max"][:, :MTN], ctx_off["global_max"][:, :MTN])
    assert torch.equal(ctx_on["global_sum"][:, :MTN], ctx_off["global_sum"][:, :MTN])


def test_flag_is_identity_at_nl1():
    """T <= mtn: a single level and no summaries, so the flag changes nothing."""
    q, k, v = _inputs(MTN, 8, 2, torch.bfloat16, seed=3100)
    with torch.no_grad():
        for kw in (dict(), dict(_save_for_backward=True, _save_parents=True)):
            a = recsa_forward(q, k, v, compression_rate=CR, max_top_nodes=MTN, top_k_per_layer=TK,
                              rope_base=ROPE, no_summaries=True, **kw)
            b = recsa_forward(q, k, v, compression_rate=CR, max_top_nodes=MTN, top_k_per_layer=TK,
                              rope_base=ROPE, no_summaries=False, **kw)
            a = a[0] if isinstance(a, tuple) else a
            b = b[0] if isinstance(b, tuple) else b
            assert torch.equal(a, b)


# ---------------------------------------------------------- non-degeneracy

@pytest.mark.parametrize("T,expect_above_tol", [
    pytest.param(T_NL2, False, id="nl2"),
    pytest.param(T_NL3, True, id="nl3"),    # ~3,840 dropped mid nodes per row: large mass
])
@pytest.mark.parametrize("path", list(PATHS))
def test_flag_is_not_degenerate(T, expect_above_tol, path, record_m1_result):
    q, k, v = _inputs(T, 8, 2, torch.float32, seed=4000 + T, v_scale=8.0)
    on, _ = _fwd(q, k, v, path, no_summaries=True)
    off, _ = _fwd(q, k, v, path, no_summaries=False)
    late_on, late_off = on[:, MTN:].float(), off[:, MTN:].float()
    d = (late_on - late_off).abs()
    rel = (late_on - late_off).norm().item() / late_off.norm().item()
    rows_diff = (d.amax(dim=(2, 3)) > 0).float().mean().item()
    print(f"[no-summaries non-degeneracy] {path} T={T}: max|d|={d.max().item():.4g} rel_L2={rel:.4g} "
          f"rows changed={rows_diff:.4f}")
    record_m1_result({"check": "a2_non_degeneracy", "path": path, "T": T,
                      "max_abs": d.max().item(), "rel_l2": rel, "rows_changed_frac": rows_diff})
    assert not torch.equal(late_on, late_off), "the flag changed nothing — degenerate"
    assert rows_diff > 0.5, f"only {rows_diff:.3f} of the rows with summaries changed"
    if expect_above_tol:
        assert d.max().item() > 1e-2 and rel > 1e-2, (d.max().item(), rel)


# ------------------------------------------------------ prefill paths

@pytest.mark.parametrize("T,dtype", [
    pytest.param(T_NL2, torch.float32, id="nl2-fp32"),
    pytest.param(T_NL2, torch.bfloat16, id="nl2-bf16"),
    pytest.param(T_NL3, torch.bfloat16, id="nl3-bf16"),
])
def test_prefill_paths_bitwise_with_flag_on(T, dtype, record_m1_result):
    q, k, v = _inputs(T, 8, 2, dtype, seed=5000 + T)
    res = {p: _fwd(q, k, v, p, no_summaries=True) for p in PATHS}
    out_ref, ctx_ref = res["train"]
    nl = ctx_ref["num_layers"]
    for p in ("ns", "legacy_top"):
        out, ctx = res[p]
        for li in range(nl):
            assert torch.equal(ctx["per_layer_parents"][li], ctx_ref["per_layer_parents"][li]), (p, li)
        assert torch.equal(out, out_ref), (
            f"{p}: {(out != out_ref).sum().item()} elements differ with identical selection")
    out_s, ctx_s = res["eval_split"]
    same_rows = torch.ones(T, dtype=torch.bool, device="cuda")
    for li in range(nl):
        same_rows &= (ctx_s["per_layer_parents"][li] == ctx_ref["per_layer_parents"][li]).flatten(2).all(-1)[0]
    n_tie = int((~same_rows).sum())
    print(f"[no-summaries bitwise] T={T} {dtype}: split rows with a tie-collapsed selection: {n_tie}/{T}")
    record_m1_result({"check": "a2_bitwise_bonus", "T": T, "dtype": str(dtype),
                      "split_rows_tie_collapsed": n_tie})
    assert torch.equal(out_s[:, same_rows], out_ref[:, same_rows])
