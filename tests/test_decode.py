"""Decode session on a tiny grafted Qwen3: decode steps against fresh prefills, tree exactness and CUDA graphs."""
import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from transformers import Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM  # noqa: E402

from graft import install_recsa_attention, install_streamed_forward  # noqa: E402
from recsa.decode import RecSADecodeSession  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _deployed_prefill_pin(deployed_prefill):
    """Runs this module on the reference prefill (every RECSA_PREFILL_FAST path off)."""

TK, MTN = 64, 1024
C_RATE = 16


def _model():
    cfg = Qwen3Config(
        hidden_size=512, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=128, intermediate_size=1024,
        vocab_size=512, max_position_embeddings=65536, rope_theta=5e6,
        use_sliding_window=False, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def _graft(m, tmp_path=None, shared_probe=False, q_chunk=4096,
           deployed_strength=False):
    kw = dict(mode="internal", rope_base=5e6, top_k=TK, max_top_nodes=MTN)
    if shared_probe:
        torch.manual_seed(3)
        if deployed_strength:
            w = torch.randn(2, 128)
            w = w / w.norm(dim=-1, keepdim=True) * 12.35   # deployed ||w||
            ck = {"probe_w": w,
                  "probe_tau": torch.full((2,), 0.7717),   # deployed tau
                  "probe_gate": torch.full((2,), 1.0),     # deployed gate
                  "probe_mode": "shared_std"}
        else:
            ck = {"probe_w": torch.randn(2, 128) * 0.05,
                  "probe_tau": torch.full((2,), 0.05),
                  "probe_gate": torch.full((2,), 0.9),
                  "probe_mode": "shared_std"}
        p = tmp_path / "probe_shared.pt"
        torch.save(ck, p)
        kw["probe_ckpt"] = str(p)
    install_recsa_attention(m, **kw)
    install_streamed_forward(m, q_chunk=q_chunk)
    return m


def _gd3_ladder(m, t0, n, atol=None):
    torch.manual_seed(42)
    ids = torch.randint(0, 512, (1, t0 + n), device="cuda")
    full = m(ids, logits_to_keep=n + 1).logits[0].float()  # rows t0-1 .. t0+n-1
    sess = RecSADecodeSession(m, reserve=n + 8)
    rows = [sess.prefill(ids[:, :t0])[0]]
    for i in range(n):
        rows.append(sess.step(ids[:, t0 + i:t0 + i + 1])[0])
    got = torch.stack(rows)
    assert got.shape == full.shape
    if atol is None:
        assert torch.equal(got, full), \
            f"max |d|={ (got - full).abs().max().item() }"
    else:
        assert torch.allclose(got, full, atol=atol, rtol=0), \
            f"max |d|={ (got - full).abs().max().item() }"
        _argmax_agree(got, full, atol)
    return sess


def _argmax_agree(got, full, atol, max_flip_frac=0.1):
    mism = got.argmax(-1) != full.argmax(-1)
    if mism.any():
        top2 = full.topk(2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        assert (margin[mism] <= 2 * atol).all(), \
            f"argmax flip with margin {margin[mism].max().item():.3f} > noise"
        assert mism.float().mean().item() <= max_flip_frac


def test_kernel_bucket_build_matches_full():
    from recsa.parallel import recsa_build_pyramid
    g = torch.Generator(device="cuda").manual_seed(11)
    for w in (16385, 16000, 1025, 993, *range(1026, 1040)):
        k = torch.randn(1, w, 2, 128, generator=g, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, w, 2, 128, generator=g, device="cuda", dtype=torch.bfloat16)
        with torch.cuda.device(0):
            fk, fv = recsa_build_pyramid(k, v, 16, max(1, (w + 15) // 16))
        for a in range(0, w, 16):
            b = min(a + 16, w)
            if b - a == 1:
                pk, pv = k[:, a:b], v[:, a:b]
            else:
                with torch.cuda.device(0):
                    ck, cv = recsa_build_pyramid(
                        k[:, a:b].contiguous(), v[:, a:b].contiguous(), 16, 1)
                pk, pv = ck[1], cv[1]
            p = a // 16
            assert torch.equal(pk[0, 0], fk[1][0, p]) and torch.equal(pv[0, 0], fv[1][0, p]), \
                f"w={w} parent {p}"


def test_fused_probe_pool_matches_bucket_pool():
    """recsa_probe_pool_kernel against its torch reference DecodeBucketPool."""
    from recsa.parallel import probe_pool_bucket
    from recsa.probe_pool import DecodeBucketPool
    H, D, C = 8, 128, 16
    torch.manual_seed(5)
    w_dep = torch.randn(D, device="cuda")
    w_dep = (w_dep / w_dep.norm() * 12.35).float()
    w_med = torch.randn(D, device="cuda").float()
    w_weak = (torch.randn(D, device="cuda") * 0.05).float()

    def _t(x):
        return None if x is None else torch.tensor(float(x), device="cuda")

    regimes = [
        ("deployed", w_dep, _t(0.7717), _t(1.0)),
        ("medium", w_med, _t(0.3), _t(0.7)),
        ("weak", w_weak, _t(0.05), _t(0.9)),
        ("nogate", w_dep, _t(0.7717), None),
    ]
    g = torch.Generator(device="cuda").manual_seed(7)

    def _datasets(n):
        base = torch.randn(1, n, H, D, generator=g, device="cuda", dtype=torch.bfloat16)
        one = torch.randn(1, 1, H, D, generator=g, device="cuda", dtype=torch.bfloat16)
        return [("randn", base), ("identical", one.expand(1, n, H, D).contiguous()),
                ("big", base * 1e4), ("tiny", base * 2.0 ** -13)]

    ref_pool = DecodeBucketPool(H, D, C, torch.device("cuda"))
    bf16_mism = 0
    worst_fp32 = 0.0
    for name, w, tau, gate in regimes:
        for n in range(1, C + 1):
            for dname, ck in _datasets(n):
                cv = torch.randn(1, n, H, D, generator=g, device="cuda",
                                 dtype=torch.bfloat16)
                rk, rv = ref_pool(ck, cv, w, tau, gate)  # fp32 [1,1,H,D]
                if name == "deployed" and dname == "randn" and n >= 4:
                    mk, _ = ref_pool(ck, cv, w, torch.zeros_like(tau), gate)
                    assert (rk - mk).abs().max().item() > 0.05, (
                        f"n={n}: deployed-strength probe within 0.05 of mean "
                        "pooling — fixture degenerated, gate proves nothing")
                for out_dtype in (torch.float32, torch.bfloat16):
                    ok = torch.zeros(1, 1, H, D, device="cuda", dtype=out_dtype)
                    ov = torch.zeros(1, 1, H, D, device="cuda", dtype=out_dtype)
                    with torch.cuda.device(0):
                        probe_pool_bucket(ck, cv, w, tau, gate, ok, ov, C)
                    if out_dtype is torch.float32:
                        scale = max(1.0, rk.abs().max().item())
                        dk = (ok - rk).abs().max().item()
                        dv = (ov - rv).abs().max().item()
                        worst_fp32 = max(worst_fp32, dk / scale, dv / scale)
                        assert dk <= 3e-5 * scale and dv <= 3e-5 * scale, (
                            f"{name}/{dname} n={n}: fp32 maxd k={dk:.2e} "
                            f"v={dv:.2e} scale={scale:.1e}")
                    else:
                        ek, ev = rk.to(torch.bfloat16), rv.to(torch.bfloat16)
                        scale = max(1.0, rk.abs().max().item())
                        for got, exp, which in ((ok, ek, "K"), (ov, ev, "V")):
                            d = (got.float() - exp.float()).abs()
                            m = int((d > 0).sum())
                            bf16_mism += m
                            if m:
                                cell = exp.float().abs() * 2.0 ** -7  # bf16: 7 stored mantissa bits
                                bad = d > cell + 3e-5 * scale
                                assert not bool(bad.any()), (
                                    f"{name}/{dname} n={n} {which}: bf16 delta "
                                    f"{d[bad].max().item():.2e} beyond cell+"
                                    f"noise at {int(bad.sum())} elems")
    print(f"fused pool vs reference: worst fp32 rel {worst_fp32:.2e}, "
          f"bf16 mismatched elems (<=1 ulp) {bf16_mism}")


@pytest.mark.parametrize("fused", [False, True], ids=["torchpool", "fusedpool"])
def test_production_shape_tree_parity(fused, monkeypatch):
    """Tree exactness at the deployed shape and length, without a model."""
    import recsa.incremental_tree as itree
    from recsa.incremental_tree import IncrementalTreeState
    from recsa.parallel import recsa_build_pyramid
    from recsa.probe_pool import build_tree_layers
    monkeypatch.setattr(itree, "FUSED_POOL_DEFAULT", fused)
    T0, N = 131050, 128
    g = torch.Generator(device="cuda").manual_seed(20260730)
    k = torch.randn(1, T0 + N, 8, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, T0 + N, 8, 128, generator=g, device="cuda", dtype=torch.bfloat16)

    with torch.cuda.device(0):
        pk, pv = recsa_build_pyramid(k[:, :T0].contiguous(), v[:, :T0].contiguous(), 16, MTN)
    st = IncrementalTreeState.from_layers(pk, pv, 16, MTN, reserve=N + 8)
    assert st.num_layers == 3, st.num_layers
    for t in range(T0, T0 + N):
        st.append(k[:, t], v[:, t])
    del pk, pv
    with torch.cuda.device(0):
        rk, rv = recsa_build_pyramid(k.contiguous(), v.contiguous(), 16, MTN)
    gk, gv = st.layers()
    assert [x.shape[1] for x in gk] == [x.shape[1] for x in rk]
    for lvl in range(len(rk)):
        assert torch.equal(gk[lvl], rk[lvl]), f"mean level {lvl} K not bitwise"
        assert torch.equal(gv[lvl], rv[lvl]), f"mean level {lvl} V not bitwise"
    del st, rk, rv, gk, gv
    torch.cuda.empty_cache()

    torch.manual_seed(3)
    pw = (torch.randn(128, device="cuda") * 0.05).float()
    pt = torch.tensor(0.05, device="cuda")
    pg = torch.tensor(0.9, device="cuda")
    kw = dict(probe_w=pw, probe_tau=pt, probe_mode="shared_std", probe_gate=pg)
    lk, lv = build_tree_layers(k[:, :T0], v[:, :T0], 16, MTN, **kw)
    stp = IncrementalTreeState.from_layers(lk, lv, 16, MTN, reserve=N + 8, **kw)
    del lk, lv
    for t in range(T0, T0 + N):
        stp.append(k[:, t], v[:, t])
    fk, fv = build_tree_layers(k, v, 16, MTN, **kw)
    ak, av = stp.layers()
    worst = []
    for lvl in range(len(fk)):
        if fused and lvl > 0:
            dvv = (av[lvl].float() - fv[lvl].float()).abs()
            nv = int((dvv > 0).sum())
            if nv:
                cell = fv[lvl].float().abs() * 2.0 ** -7  # bf16: 7 stored mantissa bits
                sc = max(1.0, fv[lvl].float().abs().max().item())
                bad = dvv > cell + 3e-5 * sc
                assert not bool(bad.any()), (
                    f"probe level {lvl} V (fused): delta beyond cell+noise "
                    f"at {int(bad.sum())} elems, max {dvv[bad].max().item():.2e}")
            print(f"fused V level {lvl}: {nv} elems differ")
        else:
            assert torch.equal(av[lvl], fv[lvl]), f"probe level {lvl} V not bitwise"
        if lvl == 0:
            assert torch.equal(ak[lvl], fk[lvl]), "probe leaves must be bitwise"
            continue
        d = (ak[lvl].float() - fk[lvl].float()).abs()
        rel = (d / fk[lvl].float().abs().clamp_min(1e-3)).max().item()
        frac = (d > 0).float().mean().item()
        worst.append((lvl, rel, frac))
        assert rel <= 2e-2, f"probe level {lvl} K rel {rel} (frac {frac})"
        assert frac <= 0.10, f"probe level {lvl} K differs on {frac:.3f} of parents"
        if not fused:
            assert frac == 0.0, (
                f"probe level {lvl} K is expected bitwise at T={T0}+{N} and "
                f"now differs on {frac:.4f} of parents (max rel {rel:.2e}) — still "
                f"inside the open-bucket tolerance, but the reduction strategy changed")
        else:
            assert frac <= 1e-5, (
                f"probe level {lvl} K (fused) differs on {frac:.2e} of parents "
                f"(expected about 2.4e-07 — the fused kernel's noise "
                f"band moved)")
    print("probe K vs batch build (level, max_rel, frac_differing):", worst)


DEPLOYED_PROBE = Path(os.environ.get("RECSA_TEST_ARTIFACTS", REPO / "test_artifacts")) / "stage1a_shared_init_probe.pt"


@pytest.mark.skipif(not DEPLOYED_PROBE.exists(), reason="deployed probe ckpt absent")
@pytest.mark.parametrize("fused", [False, True], ids=["torchpool", "fusedpool"])
def test_production_shape_tree_parity_deployed_probe(fused, monkeypatch):
    import recsa.incremental_tree as itree
    from recsa.incremental_tree import IncrementalTreeState
    from recsa.probe_pool import _pool_one_level, build_tree_layers
    monkeypatch.setattr(itree, "FUSED_POOL_DEFAULT", fused)
    ck = torch.load(DEPLOYED_PROBE, map_location="cpu", weights_only=False)
    assert ck.get("probe_mode") == "shared_std", ck.get("probe_mode")
    w_all, tau_all, g_all = ck["probe_w"].float(), ck["probe_tau"].float(), ck["probe_gate"].float()
    spread = tau_all * w_all.norm(dim=-1)
    li = int(spread.argmax())  # worst layer: strongest selection pressure
    pw = w_all[li].cuda()
    pt = tau_all[li].cuda()
    pg = g_all[li].cuda()
    print(f"deployed layer {li}: tau*||w||={spread[li]:.3f} (median {spread.median():.3f}), "
          f"gate={float(pg):.3f}")

    T0, N = 131050, 128
    g = torch.Generator(device="cuda").manual_seed(20260730)
    k = torch.randn(1, T0 + N, 8, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, T0 + N, 8, 128, generator=g, device="cuda", dtype=torch.bfloat16)

    kf = k[:, :4096].float().reshape(1, 256, 16, 8, 128)
    mu = kf.mean(dim=2, keepdim=True)
    dev = kf - mu
    sigma = (dev.pow(2).sum(dim=(2, 4)) / (16 * 128)).clamp_min(1e-12).sqrt()
    logits = pt * (dev @ pw) / sigma[:, :, None, :]
    alpha = torch.softmax(logits, dim=2)
    a_max = alpha.max(dim=2).values.mean().item()
    print(f"alpha_max mean={a_max:.3f} (uniform would be {1 / 16:.4f})")
    assert a_max > 0.25, (
        f"alpha_max {a_max:.3f} — the operator degenerated to mean pooling, so "
        "this gate would not be testing the probe at all")

    kw = dict(probe_w=pw, probe_tau=pt, probe_mode="shared_std", probe_gate=pg)
    lk, lv = build_tree_layers(k[:, :T0], v[:, :T0], 16, MTN, **kw)
    st = IncrementalTreeState.from_layers(lk, lv, 16, MTN, reserve=N + 8, **kw)
    assert st.num_layers == 3
    del lk, lv
    for t in range(T0, T0 + N):
        st.append(k[:, t], v[:, t])
    fk, fv = build_tree_layers(k, v, 16, MTN, **kw)
    ak, av = st.layers()
    ULP_BOUND = {1: 4, 2: 48}
    win = {}
    for lvl in (1, 2):
        div = C_RATE ** (lvl - 1)
        win[lvl] = ((T0 // div - 1) // C_RATE, ((T0 + N) // div - 1) // C_RATE)
    report = []
    for lvl in range(len(fk)):
        if fused and lvl > 0:
            dvv = (av[lvl].float() - fv[lvl].float()).abs()
            nv = int((dvv > 0).sum())
            if nv:
                vcell = fv[lvl].float().abs() * 2.0 ** -7  # bf16: 7 stored mantissa bits
                vsc = max(1.0, fv[lvl].float().abs().max().item())
                vbad = dvv > vcell + 3e-5 * vsc
                assert not bool(vbad.any()), (
                    f"level {lvl} V (fused): delta beyond cell+noise at "
                    f"{int(vbad.sum())} elems, max {dvv[vbad].max().item():.2e}")
                vidx = (dvv.sum(dim=(0, 2, 3)) > 0).nonzero().flatten().tolist()
                lo_v, hi_v = win[lvl]
                assert all(lo_v <= i <= hi_v for i in vidx), (
                    f"level {lvl} V (fused): parents outside the recompute "
                    f"window differ — FROZEN HISTORY MOVED")
            print(f"fused V level {lvl}: {nv} elems differ")
        else:
            assert torch.equal(av[lvl], fv[lvl]), f"level {lvl} V not bitwise (V is mean-pooled)"
        if lvl == 0:
            assert torch.equal(ak[lvl], fk[lvl]), "leaves must be bitwise"
            continue
        a, f = ak[lvl].float(), fk[lvl].float()
        d = (a - f).abs()
        n = int((d > 0).sum())
        ulp = torch.where(f.abs() > 0, f.abs().log2().floor().exp2() * 2 ** -8,
                          torch.full_like(f, 2 ** -126))
        u = (d / ulp).max().item() if n else 0.0
        idx = (d.sum(dim=(0, 2, 3)) > 0).nonzero().flatten().tolist()
        report.append((lvl, n, round(u, 1), (min(idx), max(idx)) if idx else None))
        lo, hi = win[lvl]
        assert all(lo <= i <= hi for i in idx), (
            f"level {lvl}: parents {[i for i in idx if not lo <= i <= hi]} differ but "
            f"were never recomputed by this session (window {lo}..{hi}) — FROZEN "
            f"HISTORY MOVED, which no tolerance covers")
        assert u <= ULP_BOUND[lvl], (
            f"level {lvl}: {u} ulps > {ULP_BOUND[lvl]} — the open-bucket reduction-order "
            f"effect does not reach this magnitude; suspect an alpha child flip")
    print("deployed-probe K vs batch build (level, n_diff, max_ulps, parent_range):", report)
    child, cv = k[:, :4096], v[:, :4096]
    full_p, _ = _pool_one_level(child, cv, 16, pw, pt, standardize=True, probe_gate=pg)

    def _alpha(x):
        kf = x.float().reshape(1, x.shape[1] // 16, 16, 8, 128)
        mu = kf.mean(dim=2, keepdim=True)
        dv = kf - mu
        sg = (dv.pow(2).sum(dim=(2, 4)) / (16 * 128)).clamp_min(1e-12).sqrt()
        return torch.softmax(pt * (dv @ pw) / sg[:, :, None, :], dim=2)

    win_full = _alpha(child).argmax(dim=2)
    for b in range(0, child.shape[1] // 16, 37):
        sl = slice(b * 16, (b + 1) * 16)
        one_p, _ = _pool_one_level(child[:, sl], cv[:, sl], 16, pw, pt,
                                   standardize=True, probe_gate=pg)
        assert torch.equal(_alpha(child[:, sl]).argmax(dim=2)[:, 0], win_full[:, b]), (
            f"bucket {b}: the alpha winner CHANGED between a 1-bucket recompute and "
            f"the full-level build — a discrete child swap, not a rounding effect")
        d = (one_p[:, 0].float() - full_p[:, b].float()).abs()
        f = full_p[:, b].float()
        ulp = torch.where(f.abs() > 0, f.abs().log2().floor().exp2() * 2 ** -8,
                          torch.full_like(f, 2 ** -126))
        assert (d / ulp).max().item() <= 4, f"bucket {b}: {(d / ulp).max().item()} ulps"


def test_gd3_mean_bitwise_per_prefix_depth_stable():
    m = _graft(_model())
    t0, n = 16000, 40
    torch.manual_seed(41)
    ids = torch.randint(0, 512, (1, t0 + n), device="cuda")
    sess = RecSADecodeSession(m, reserve=n + 4)
    row = sess.prefill(ids[:, :t0])
    assert torch.equal(row, m(ids[:, :t0], logits_to_keep=1).logits[:, -1].float())
    for i in range(n):
        row = sess.step(ids[:, t0 + i:t0 + i + 1])
        ref = m(ids[:, :t0 + i + 1], logits_to_keep=1).logits[:, -1].float()
        assert torch.equal(row, ref), \
            f"step {i}: max |d|={(row - ref).abs().max().item()}"
    assert sess.states[0].num_layers == 2 and sess.t == t0 + n


def test_gd3_multirow_reference_close():
    m = _graft(_model())
    _gd3_ladder(m, 16300, 100, atol=2.5e-1)


def test_gd3_level_growth_crossing_semantics():
    m = _graft(_model())
    t0, n = 16370, 30
    torch.manual_seed(43)
    ids = torch.randint(0, 512, (1, t0 + n), device="cuda")
    sess = RecSADecodeSession(m, reserve=n + 4)
    row = sess.prefill(ids[:, :t0])
    assert torch.equal(row, m(ids[:, :t0], logits_to_keep=1).logits[:, -1].float())
    boundary = 16385  # first t+1 with nl=3
    for i in range(n):
        row = sess.step(ids[:, t0 + i:t0 + i + 1])
        ref = m(ids[:, :t0 + i + 1], logits_to_keep=1).logits[:, -1].float()
        if t0 + i + 1 < boundary:
            assert torch.equal(row, ref), \
                f"step {i} (pre-boundary): max |d|={(row - ref).abs().max().item()}"
        else:
            d = (row - ref).abs().max().item()
            assert d <= 2.5e-1, f"step {i}: max |d|={d}"
            _argmax_agree(row, ref, 2.5e-1, max_flip_frac=1.0)
    assert sess.states[0].num_layers == 3 and sess.t == t0 + n
    fresh = RecSADecodeSession(m, reserve=4)
    fresh.prefill(ids)
    ak, av = sess.states[0].layers()
    bk, bv = fresh.states[0].layers()
    for x, y in zip(ak + av, bk + bv):
        assert torch.equal(x, y), "layer-0 tree maintenance diverged"


def test_gd3_probe_shared_std_close(tmp_path):
    m = _graft(_model(), tmp_path, shared_probe=True)
    _gd3_ladder(m, 4000, 48, atol=2e-2)


def test_gd3_probe_shared_std_nl3_close(tmp_path):
    m = _graft(_model(), tmp_path, shared_probe=True)
    t0, n = 16390, 24
    torch.manual_seed(46)
    ids = torch.randint(0, 512, (1, t0 + n), device="cuda")
    sess = RecSADecodeSession(m, reserve=n + 4)
    row = sess.prefill(ids[:, :t0])
    assert sess.states[0].num_layers == 3
    assert torch.equal(row, m(ids[:, :t0], logits_to_keep=1).logits[:, -1].float()), \
        "probe prefill must BE the reference prefill (same builder, same call)"
    d = []
    for i in range(n):
        row = sess.step(ids[:, t0 + i:t0 + i + 1])
        ref = m(ids[:, :t0 + i + 1], logits_to_keep=1).logits[:, -1].float()
        d.append((row - ref).abs().max().item())
        _argmax_agree(row, ref, 5e-2)
    assert max(d) <= 5e-2, f"NL=3 shared_std per-prefix drift {max(d)}"
    head, tail = sum(d[:6]) / 6, sum(d[-6:]) / 6
    assert tail <= 3 * head + 1e-3, f"drift grows: head {head} -> tail {tail}"


def test_gd3_full_generation_length():
    """128 steps, the eval's longest generation, in the deployed NL=3 band."""
    m = _graft(_model())
    t0, n = 16390, 128  # 16,518 < 262,144: NL=3 throughout
    torch.manual_seed(47)
    ids = torch.randint(0, 512, (1, t0 + n), device="cuda")
    sess = RecSADecodeSession(m, reserve=n + 4)
    row = sess.prefill(ids[:, :t0])
    assert sess.states[0].num_layers == 3
    assert torch.equal(row, m(ids[:, :t0], logits_to_keep=1).logits[:, -1].float()), \
        "prefill must be bitwise: both sides are the same call"
    d, n_exact = [], 0
    for i in range(n):
        row = sess.step(ids[:, t0 + i:t0 + i + 1])
        ref = m(ids[:, :t0 + i + 1], logits_to_keep=1).logits[:, -1].float()
        dm = (row - ref).abs().max().item()
        d.append(dm)
        n_exact += dm == 0.0
        assert dm <= 5e-2, f"step {i}/{n}: max |d|={dm} — beyond GEMM-shape wobble"
        _argmax_agree(row, ref, 5e-2, max_flip_frac=0.0 if dm == 0 else 1.0)
    head, tail = sum(d[:16]) / 16, sum(d[-16:]) / 16
    assert tail <= 3 * head + 1e-3, f"drift GROWS over 128 steps: {head} -> {tail}"
    print(f"128-step ladder: {n_exact}/{n} steps bitwise, max |d| {max(d):.4g}")
    assert sess.t == t0 + n and sess.states[0].num_layers == 3
    fresh = RecSADecodeSession(m, reserve=4)
    fresh.prefill(ids)
    ak, av = sess.states[0].layers()
    bk, bv = fresh.states[0].layers()
    for lvl, (x, y) in enumerate(zip(ak + av, bk + bv)):
        assert torch.equal(x, y), f"level {lvl % len(ak)} diverged after {n} appends"


def test_gd3_bitwise_resumes_past_boundary():
    m = _graft(_model())
    t0, n = 16390, 6
    torch.manual_seed(44)
    ids = torch.randint(0, 512, (1, t0 + n), device="cuda")
    sess = RecSADecodeSession(m, reserve=n + 2)
    row = sess.prefill(ids[:, :t0])
    assert sess.states[0].num_layers == 3
    assert torch.equal(row, m(ids[:, :t0], logits_to_keep=1).logits[:, -1].float())
    for i in range(n):
        row = sess.step(ids[:, t0 + i:t0 + i + 1])
        ref = m(ids[:, :t0 + i + 1], logits_to_keep=1).logits[:, -1].float()
        assert torch.equal(row, ref), f"step {i}"


def test_gemm_mshape_canary_tiny():
    """At the tiny test shapes, bf16 linear rows must not depend on batch M."""
    torch.manual_seed(12)
    x_full = torch.randn(8192, 9728, dtype=torch.bfloat16, device="cuda")

    lin = torch.nn.Linear(512, 512, bias=False, dtype=torch.bfloat16, device="cuda")
    x = x_full[:4096, :512].contiguous()
    full = lin(x)
    for m_rows in (1, 16, 33, 201):
        assert torch.equal(lin(x[:m_rows]), full[:m_rows]), f"K=512 M={m_rows}"

    sensitive = {}
    for k_dim in (1024, 2560, 4096, 9728):
        lk = torch.nn.Linear(k_dim, 512, bias=False, dtype=torch.bfloat16,
                             device="cuda")
        xk = x_full[:, :k_dim].contiguous()
        ref = lk(xk)
        worst = 0.0
        for m_rows in (1, 2, 33, 96, 512, 1000):
            d = (lk(xk[:m_rows]).float() - ref[:m_rows].float()).abs().max()
            worst = max(worst, float(d))
        sensitive[k_dim] = worst
        if k_dim <= 2560:
            assert worst == 0.0, (
                f"K={k_dim} is no longer M-shape invariant (max|d|={worst:.3g}) "
                "— the split-K threshold moved DOWN; fixture-width bitwise "
                "gates are no longer sound")
    assert sensitive[9728] > 0.0, (
        f"K=9728 is now M-shape invariant (map={sensitive}) — this canary can "
        "no longer see split-K, so it certifies nothing about the deployed "
        "model; re-derive the threshold before trusting any cross-shape gate")


def test_step_guards_poison_and_state_count():
    m = _graft(_model())
    torch.manual_seed(13)
    ids = torch.randint(0, 512, (1, 2048), device="cuda")
    sess = RecSADecodeSession(m, reserve=8)
    sess.prefill(ids)
    tok = torch.tensor([[7]], device="cuda")
    boom = RuntimeError("injected")
    orig = m.model.layers[1].mlp.forward

    def raiser(*a, **k):
        raise boom

    m.model.layers[1].mlp.forward = raiser
    try:
        with pytest.raises(RuntimeError, match="injected"):
            sess.step(tok)
    finally:
        m.model.layers[1].mlp.forward = orig
    with pytest.raises(RuntimeError, match="poisoned"):
        sess.step(tok)
    fresh = RecSADecodeSession(m, reserve=8)
    ref_pre = fresh.prefill(ids)
    ref_row = fresh.step(tok)
    got_pre = sess.prefill(ids)
    got_row = sess.step(tok)
    assert torch.equal(got_pre, ref_pre) and torch.equal(got_row, ref_row)
    sess2 = RecSADecodeSession(m, reserve=8)
    sess2.prefill(ids)
    sess2.states = sess2.states[:1]
    with pytest.raises(RuntimeError, match="did not complete"):
        sess2.step(tok)


def test_prefill_failure_after_spills_refuses_step():
    m = _graft(_model())
    torch.manual_seed(14)
    ids = torch.randint(0, 512, (1, 2048), device="cuda")
    sess = RecSADecodeSession(m, reserve=8)
    orig = m.lm_head.forward

    def raiser(*a, **k):
        raise RuntimeError("injected-late")

    m.lm_head.forward = raiser
    try:
        with pytest.raises(RuntimeError, match="injected-late"):
            sess.prefill(ids)
    finally:
        m.lm_head.forward = orig
    tok = torch.tensor([[7]], device="cuda")
    with pytest.raises(RuntimeError, match="did not complete"):
        sess.step(tok)
    # and the same session recovers via a successful prefill
    ref = RecSADecodeSession(m, reserve=8)
    assert torch.equal(sess.prefill(ids), ref.prefill(ids))
    assert torch.equal(sess.step(tok), ref.step(tok))


def test_step_failure_after_appends_poisons():
    m = _graft(_model())
    torch.manual_seed(15)
    ids = torch.randint(0, 512, (1, 2048), device="cuda")
    sess = RecSADecodeSession(m, reserve=8)
    sess.prefill(ids)
    tok = torch.tensor([[7]], device="cuda")
    orig = m.lm_head.forward

    def raiser(*a, **k):
        raise RuntimeError("injected-tail")

    m.lm_head.forward = raiser
    try:
        with pytest.raises(RuntimeError, match="injected-tail"):
            sess.step(tok)
    finally:
        m.lm_head.forward = orig
    with pytest.raises(RuntimeError, match="poisoned"):
        sess.step(tok)


def test_gd4_greedy_matches_teacher_forced():
    m = _graft(_model())
    torch.manual_seed(7)
    prompt = torch.randint(0, 512, (1, 2048), device="cuda")
    got = RecSADecodeSession(m, reserve=40).generate(prompt, 32)
    ids = prompt
    for _ in range(32):
        nxt = int(m(ids, logits_to_keep=1).logits[0, -1].float().argmax(-1))
        ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda")], 1)
    assert got == ids[0, 2048:].tolist()


def test_gd4_greedy_probe_deployed_strength_nl3(tmp_path):
    m = _graft(_model(), tmp_path, shared_probe=True, deployed_strength=True)
    t0, n = 16390, 128
    torch.manual_seed(48)
    prompt = torch.randint(0, 512, (1, t0), device="cuda")
    sess = RecSADecodeSession(m, reserve=n + 8)
    got = sess.generate(prompt, n)
    assert sess.states[0].num_layers == 3
    ids = prompt
    ref = []
    for i in range(n):
        nxt = int(m(ids, logits_to_keep=1).logits[0, -1].float().argmax(-1))
        ref.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda")], 1)
        if nxt != got[i]:
            raise AssertionError(
                f"probe greedy trajectory diverged at step {i}/{n}: session "
                f"{got[i]} vs teacher-forced {nxt}. Every later token is forked, "
                f"so the evaluation's string for this sample would differ.")
    assert got == ref


def test_generate_respects_reserve_and_eos():
    m = _graft(_model())
    torch.manual_seed(9)
    prompt = torch.randint(0, 512, (1, 1500), device="cuda")
    sess = RecSADecodeSession(m, reserve=8)
    with pytest.raises(ValueError, match="reserve"):
        sess.generate(prompt, 64)
    first = sess.generate(prompt, 1)
    sess2 = RecSADecodeSession(m, reserve=8)
    assert sess2.generate(prompt, 4, eos_token_ids=[first[0]]) == first
