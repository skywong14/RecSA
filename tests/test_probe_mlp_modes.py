"""MLP probe modes (mlp_scorer / mlp_pooler)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from recsa.incremental_tree import IncrementalTreeState  # noqa: E402
from recsa.probe_pool import build_tree_layers  # noqa: E402

C, MTN, HKV, D, H1 = 16, 256, 4, 128, 32
T = 5003  # ragged at every level: 5003 -> 313 -> 20


def _params(mode, *, w2_zero, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(D, generator=g, dtype=dtype) * D ** -0.5
    tau = torch.tensor(0.4, dtype=dtype)
    gate = torch.tensor(1.07, dtype=dtype)
    in_dim = D if mode == "mlp_scorer" else 2 * D
    w2_shape = (H1,) if mode == "mlp_scorer" else (D, H1)
    extra = {
        "mlp_w1": torch.randn(H1, in_dim, generator=g, dtype=dtype) * in_dim ** -0.5,
        "mlp_b1": torch.randn(H1, generator=g, dtype=dtype) * 0.1,
        "mlp_w2": (torch.zeros(w2_shape, dtype=dtype) if w2_zero
                   else torch.randn(w2_shape, generator=g, dtype=dtype) * 0.05),
    }
    return w, tau, gate, extra


def _kv(dtype=torch.bfloat16, t=T, seed=7):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, t, HKV, D, generator=g).to(dtype)
    v = torch.randn(1, t, HKV, D, generator=g).to(dtype)
    return k, v


@pytest.mark.parametrize("mode", ["mlp_scorer", "mlp_pooler"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_zero_init_matches_shared_std(mode, dtype):
    k, v = _kv(dtype)
    w, tau, gate, extra = _params(mode, w2_zero=True)
    ref_k, ref_v = build_tree_layers(k, v, C, MTN, probe_w=w, probe_tau=tau,
                                     probe_mode="shared_std", probe_gate=gate)
    got_k, got_v = build_tree_layers(k, v, C, MTN, probe_w=w, probe_tau=tau,
                                     probe_mode=mode, probe_gate=gate,
                                     probe_extra=extra)
    assert len(ref_k) == len(got_k) == 3
    for lvl, (rk, gk, rv, gv) in enumerate(zip(ref_k, got_k, ref_v, got_v)):
        assert torch.equal(rk, gk), f"{mode} level {lvl} K differs at zero init"
        assert torch.equal(rv, gv), f"{mode} level {lvl} V differs at zero init"


@pytest.mark.parametrize("mode", ["mlp_scorer", "mlp_pooler"])
def test_nonzero_mlp_changes_k_not_v(mode):
    k, v = _kv()
    w, tau, gate, extra = _params(mode, w2_zero=False)
    ref_k, ref_v = build_tree_layers(k, v, C, MTN, probe_w=w, probe_tau=tau,
                                     probe_mode="shared_std", probe_gate=gate)
    got_k, got_v = build_tree_layers(k, v, C, MTN, probe_w=w, probe_tau=tau,
                                     probe_mode=mode, probe_gate=gate,
                                     probe_extra=extra)
    assert not torch.equal(ref_k[1], got_k[1]), f"{mode}: nonzero w2 changed nothing"
    for rv, gv in zip(ref_v, got_v):
        assert torch.equal(rv, gv), f"{mode}: V must stay mean-pooled"


@pytest.mark.parametrize("mode", ["mlp_scorer", "mlp_pooler"])
def test_grads_flow(mode):
    k, v = _kv(torch.float32, t=1024)
    # w2 == 0 (trainer init): gradient must still reach w2
    w, tau, gate, extra = _params(mode, w2_zero=True)
    ps = [w, tau, gate] + list(extra.values())
    for p in ps:
        p.requires_grad_(True)
    lk, _ = build_tree_layers(k, v, C, MTN, probe_w=w, probe_tau=tau,
                              probe_mode=mode, probe_gate=gate, probe_extra=extra)
    (lk[-1].square().sum() + lk[1].square().sum()).backward()
    g2 = extra["mlp_w2"].grad
    assert g2 is not None and torch.isfinite(g2).all() and float(g2.norm()) > 0, \
        f"{mode}: no gradient to zero-init mlp_w2 — saddle"
    # w2 != 0: gradient must reach w1/b1 too
    w, tau, gate, extra = _params(mode, w2_zero=False, seed=3)
    ps = [w, tau, gate] + list(extra.values())
    for p in ps:
        p.requires_grad_(True)
    lk, _ = build_tree_layers(k, v, C, MTN, probe_w=w, probe_tau=tau,
                              probe_mode=mode, probe_gate=gate, probe_extra=extra)
    (lk[-1].square().sum() + lk[1].square().sum()).backward()
    for name in ("mlp_w1", "mlp_b1", "mlp_w2"):
        g = extra[name].grad
        assert g is not None and torch.isfinite(g).all() and float(g.norm()) > 0, \
            f"{mode}: no/nonfinite grad to {name}"
    for name, p in (("probe_w", w), ("probe_tau", tau), ("probe_gate", gate)):
        assert p.grad is not None and torch.isfinite(p.grad).all(), \
            f"{mode}: backbone {name} grad broken"


@pytest.mark.parametrize("mode", ["mlp_scorer", "mlp_pooler"])
def test_incremental_append_matches_batch(mode):
    k, v = _kv(torch.bfloat16, t=4400, seed=11)
    w, tau, gate, extra = _params(mode, w2_zero=False, seed=5)
    t0 = 4111
    st = IncrementalTreeState(k[:, :t0], v[:, :t0], C, MTN,
                              probe_w=w, probe_tau=tau, probe_mode=mode,
                              probe_gate=gate, probe_extra=extra,
                              reserve=4400 - t0)
    st.append(k[:, t0:], v[:, t0:])
    fk, fv = build_tree_layers(k, v, C, MTN, probe_w=w, probe_tau=tau,
                               probe_mode=mode, probe_gate=gate, probe_extra=extra)
    ik, iv = st.layers()
    assert len(ik) == len(fk)
    for lvl, (a, b) in enumerate(zip(ik, fk)):
        assert a.shape == b.shape, f"level {lvl} width"
        assert torch.allclose(a.float(), b.float(), atol=3e-5, rtol=0), \
            f"{mode} level {lvl} K drift > 3e-5"
    for lvl, (a, b) in enumerate(zip(iv, fv)):
        assert torch.equal(a, b), f"{mode} level {lvl} V not bitwise"


@pytest.mark.parametrize("mode", ["mlp_scorer", "mlp_pooler"])
def test_degenerate_bucket_finite(mode):
    # identical children -> zero variance -> sigma clamp path, fwd + bwd finite
    k = torch.ones(1, 64, HKV, D, dtype=torch.float32)
    v = torch.ones(1, 64, HKV, D, dtype=torch.float32)
    w, tau, gate, extra = _params(mode, w2_zero=False, seed=9)
    ps = [w, tau, gate] + list(extra.values())
    for p in ps:
        p.requires_grad_(True)
    lk, lv = build_tree_layers(k, v, C, 16, probe_w=w, probe_tau=tau,
                               probe_mode=mode, probe_gate=gate, probe_extra=extra)
    assert all(torch.isfinite(t).all() for t in lk + lv)
    lk[-1].sum().backward()
    for p in ps:
        assert p.grad is None or torch.isfinite(p.grad).all()


def test_fail_loud_wiring():
    k, v = _kv(t=1024)
    w, tau, gate, extra = _params("mlp_scorer", w2_zero=True)
    with pytest.raises(ValueError, match="probe_extra"):
        build_tree_layers(k, v, C, MTN, probe_w=w, probe_tau=tau,
                          probe_mode="mlp_scorer", probe_gate=gate)
    with pytest.raises(ValueError, match="probe_extra"):
        build_tree_layers(k, v, C, MTN, probe_w=w, probe_tau=tau,
                          probe_mode="shared_std", probe_gate=gate,
                          probe_extra=extra)
    with pytest.raises(ValueError, match="build_chunk_parents"):
        build_tree_layers(k, v, C, MTN, probe_w=w, probe_tau=tau,
                          probe_mode="mlp_scorer", probe_gate=gate,
                          probe_extra=extra, build_chunk_parents=8)
    with pytest.raises(ValueError, match="probe_extra"):
        IncrementalTreeState(k, v, C, MTN, probe_w=w, probe_tau=tau,
                             probe_mode="mlp_pooler", probe_gate=gate)


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["mlp_scorer", "mlp_pooler"])
def test_bucket_pool_mlp_matches_general(mode, dtype, device):
    from recsa.probe_pool import DecodeBucketPoolMLP

    dev = torch.device(device)
    w, tau, gate, extra = _params(mode, w2_zero=False, seed=13)
    w, tau, gate = w.to(dev), tau.to(dev), gate.to(dev)
    extra = {n: t.to(dev) for n, t in extra.items()}
    pool = DecodeBucketPoolMLP(mode, HKV, D, C, dev)
    g = torch.Generator().manual_seed(21)
    for gate_arg in (gate, None):
        for n in range(C, 0, -1):
            ck = torch.randn(1, n, HKV, D, generator=g).to(dtype).to(dev)
            cv = torch.randn(1, n, HKV, D, generator=g).to(dtype).to(dev)
            ref_k, ref_v = _general_mlp_pool(ck, cv, mode, w, tau, gate_arg,
                                             extra)
            got_k, got_v = pool(ck, cv, w, tau, gate_arg, extra)
            assert torch.equal(got_k.to(dtype), ref_k), \
                f"{mode}/{dtype}/{device} width {n} gate={gate_arg is not None}: K"
            assert torch.equal(got_v.to(dtype), ref_v), \
                f"{mode}/{dtype}/{device} width {n} gate={gate_arg is not None}: V"


def _general_mlp_pool(ck, cv, mode, w, tau, gate, extra):
    from recsa.probe_pool import _pool_one_level_mlp
    return _pool_one_level_mlp(ck, cv, C, mode, w, tau, gate, extra)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["mlp_scorer", "mlp_pooler"])
def test_bucket_pool_mlp_cursor_matches_host(mode, dtype, device):
    from recsa.probe_pool import BucketCursor, DecodeBucketPoolMLP

    dev = torch.device(device)
    w, tau, gate, extra = _params(mode, w2_zero=False, seed=17)
    w, tau, gate = w.to(dev), tau.to(dev), gate.to(dev)
    extra = {n: t.to(dev) for n, t in extra.items()}
    cap = C * C + 9                    # capacity in tokens
    rows = [(cap + C ** lv - 1) // C ** lv + 1 for lv in range(3)]
    g = torch.Generator().manual_seed(23)

    def buf(n_rows):
        return torch.randn(1, n_rows, HKV, D, generator=g).to(dtype).to(dev)

    kv = [(buf(n_rows), buf(n_rows)) for n_rows in rows]
    pool = DecodeBucketPoolMLP(mode, HKV, D, C, dev)
    t_dev = torch.zeros(1, dtype=torch.int64, device=dev)
    win = BucketCursor(t_dev, C, D, cap)
    seen = {1: set(), 2: set()}
    for gate_arg in (gate, None):
        for t in [17 * j for j in range(C)] + [cap - 1]:
            t_dev.fill_(t)
            win.refresh(3)
            for level in (1, 2):
                n_child = (t + C ** (level - 1)) // C ** (level - 1)
                p = (n_child - 1) // C
                lo = p * C
                (ck, cv), (pk, pv) = kv[level - 1], kv[level]
                ref_k, ref_v = pk.clone(), pv.clone()
                hk, hv = pool(ck[:, lo:n_child], cv[:, lo:n_child], w, tau,
                              gate_arg, extra)
                ref_k[:, p] = hk[:, 0]
                ref_v[:, p] = hv[:, 0]
                gen_k, gen_v = pk.clone(), pv.clone()
                gk, gv = _general_mlp_pool(ck[:, lo:n_child], cv[:, lo:n_child],
                                           mode, w, tau, gate_arg, extra)
                gen_k[:, p] = gk[:, 0]
                gen_v[:, p] = gv[:, 0]
                got_k, got_v = pk.clone(), pv.clone()
                pool.cursor(ck, cv, win[level], w, tau, gate_arg, extra,
                            got_k, got_v)
                tag = f"{mode}/{dtype}/{device} t={t} level {level} n={n_child - lo}"
                assert not torch.equal(ref_k, pk), f"{tag}: vacuous slot write"
                assert torch.equal(got_k, ref_k), f"{tag}: K"
                assert torch.equal(got_v, ref_v), f"{tag}: V"
                assert torch.equal(got_k, gen_k) and torch.equal(got_v, gen_v), \
                    f"{tag}: vs the general _pool_one_level_mlp"
                seen[level].add(n_child - lo)
    assert seen[1] == seen[2] == set(range(1, C + 1)), seen


def test_bucket_pool_mlp_rejects_non_mlp_mode():
    from recsa.probe_pool import DecodeBucketPoolMLP
    with pytest.raises(ValueError, match="not an MLP probe mode"):
        DecodeBucketPoolMLP("shared_std", HKV, D, C, torch.device("cpu"))
