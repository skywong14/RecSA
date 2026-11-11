"""IncrementalTreeState parity tests."""
import pytest
import torch

from recsa.incremental_tree import IncrementalTreeState
from recsa.probe_pool import build_tree_layers

CR, MTN = 16, 64
PROBE_ATOL = 3e-5


def _mk(T, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, T, 2, 8, generator=g, dtype=dtype)
    v = torch.randn(1, T, 2, 8, generator=g, dtype=dtype)
    return k, v


def _probe(seed=1):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(8, generator=g, dtype=torch.float32)
    return w, torch.tensor(0.7), torch.tensor(0.9)


def _run_ladder(k, v, t0, checkpoints, *, exact_k, **probe_kw):
    state = IncrementalTreeState(
        k[:, :t0], v[:, :t0], CR, MTN, reserve=k.shape[1] - t0, **probe_kw
    )
    for t in range(t0, k.shape[1]):
        state.append(k[:, t], v[:, t])
        if t + 1 not in checkpoints:
            continue
        ref_k, ref_v = build_tree_layers(k[:, : t + 1], v[:, : t + 1], CR, MTN, **probe_kw)
        got_k, got_v = state.layers()
        assert len(got_k) == len(ref_k), f"T={t + 1}: {len(got_k)} vs {len(ref_k)} levels"
        for lvl, (gk, rk, gv, rv) in enumerate(zip(got_k, ref_k, got_v, ref_v)):
            assert torch.equal(gv, rv), f"T={t + 1} level {lvl}: V not bitwise"
            if exact_k:
                assert torch.equal(gk, rk), f"T={t + 1} level {lvl}: K not bitwise"
            else:
                d = (gk.float() - rk.float()).abs().max().item()
                assert d <= PROBE_ATOL, f"T={t + 1} level {lvl}: K delta {d}"
    return state


def test_mean_incremental_bitwise():
    k, v = _mk(1200)
    # 1024->1025 is the num_layers 2->3 growth at mtn=64
    cps = {101, 112, 128, 129, 512, 1023, 1024, 1025, 1026, 1040, 1200}
    state = _run_ladder(k, v, 100, cps, exact_k=True)
    assert state.num_layers == 3 and state.seq_len == 1200


def test_shared_std_incremental_close():
    k, v = _mk(1200, seed=3)
    w, tau, gate = _probe()
    _run_ladder(
        k, v, 100, {128, 1024, 1025, 1200}, exact_k=False,
        probe_w=w, probe_tau=tau, probe_mode="shared_std", probe_gate=gate,
    )


def test_per_level_incremental_close():
    k, v = _mk(400, seed=4)
    g = torch.Generator().manual_seed(5)
    w = torch.randn(4, 8, generator=g, dtype=torch.float32)
    tau = torch.full((4,), 0.5)
    _run_ladder(
        k, v, 50, {64, 65, 256, 400}, exact_k=False,
        probe_w=w, probe_tau=tau, probe_mode="per_level",
    )


def test_bf16_mean_bitwise():
    k, v = _mk(200, seed=6, dtype=torch.bfloat16)
    _run_ladder(k, v, 64, {65, 128, 200}, exact_k=True)


def test_chunk_append_matches_single_appends():
    k, v = _mk(300, seed=7)
    a = IncrementalTreeState(k[:, :250], v[:, :250], CR, MTN, reserve=50)
    b = IncrementalTreeState(k[:, :250], v[:, :250], CR, MTN, reserve=50)
    for t in range(250, 300):
        a.append(k[:, t], v[:, t])
    b.append(k[:, 250:300], v[:, 250:300])
    ak, av = a.layers()
    bk, bv = b.layers()
    for x, y in zip(ak + av, bk + bv):
        assert torch.equal(x, y)


def test_reserve_guard():
    k, v = _mk(20, seed=8)
    state = IncrementalTreeState(k[:, :16], v[:, :16], CR, MTN, reserve=3)
    state.append(k[:, 16:19], v[:, 16:19])
    with pytest.raises(RuntimeError, match="reserve exhausted"):
        state.append(k[:, 19], v[:, 19])


def test_from_layers_adopts_then_appends_bitwise():
    k, v = _mk(1200, seed=10)
    ref = IncrementalTreeState(k[:, :1100], v[:, :1100], CR, MTN, reserve=100)
    lk, lv = build_tree_layers(k[:, :1100], v[:, :1100], CR, MTN)
    st = IncrementalTreeState.from_layers(lk, lv, CR, MTN, reserve=100)
    for t in range(1100, 1200):
        ref.append(k[:, t], v[:, t])
        st.append(k[:, t], v[:, t])
    fk, fv = build_tree_layers(k, v, CR, MTN)
    for x, y, z in zip(st.layers()[0] + st.layers()[1],
                       ref.layers()[0] + ref.layers()[1], fk + fv):
        assert torch.equal(x, y) and torch.equal(x, z)


def test_from_layers_adopts_before_growth_then_crosses():
    k, v = _mk(1100, seed=12)
    lk, lv = build_tree_layers(k[:, :1020], v[:, :1020], CR, MTN)
    st = IncrementalTreeState.from_layers(lk, lv, CR, MTN, reserve=80)
    for t in range(1020, 1100):
        st.append(k[:, t], v[:, t])
    fk, fv = build_tree_layers(k, v, CR, MTN)
    assert st.num_layers == len(fk) == 3
    for x, z in zip(st.layers()[0] + st.layers()[1], fk + fv):
        assert torch.equal(x, z)

    w, tau, gate = _probe(seed=13)
    pkw = dict(probe_w=w, probe_tau=tau, probe_mode="shared_std", probe_gate=gate)
    lk, lv = build_tree_layers(k[:, :1020], v[:, :1020], CR, MTN, **pkw)
    st = IncrementalTreeState.from_layers(lk, lv, CR, MTN, reserve=80, **pkw)
    for t in range(1020, 1100):
        st.append(k[:, t], v[:, t])
    fk, fv = build_tree_layers(k, v, CR, MTN, **pkw)
    for gv, rv in zip(st.layers()[1], fv):
        assert torch.equal(gv, rv)
    for gk, rk in zip(st.layers()[0], fk):
        assert (gk.float() - rk.float()).abs().max().item() <= PROBE_ATOL


def test_append_validates_chunk_shape_and_dtype():
    k, v = _mk(40, seed=14)
    st = IncrementalTreeState(k[:, :20], v[:, :20], CR, MTN, reserve=20)
    with pytest.raises(ValueError, match="shapes differ"):
        st.append(k[:, 20:24], v[:, 20:23])
    with pytest.raises(ValueError, match="dtype"):
        st.append(k[:, 20:24].to(torch.bfloat16), v[:, 20:24].to(torch.bfloat16))
    st.append(k[:, 20:24], v[:, 20:24])  # state intact after rejections
    ref_k, ref_v = build_tree_layers(k[:, :24], v[:, :24], CR, MTN)
    for x, y in zip(st.layers()[0] + st.layers()[1], ref_k + ref_v):
        assert torch.equal(x, y)


def test_from_layers_validates_levels():
    k, v = _mk(1100, seed=11)
    lk, lv = build_tree_layers(k, v, CR, MTN)
    with pytest.raises(ValueError, match="ceil-chain"):
        IncrementalTreeState.from_layers(
            [lk[0], lk[1][:, :-1], lk[2]], [lv[0], lv[1][:, :-1], lv[2]], CR, MTN)
    with pytest.raises(ValueError, match="needs"):
        IncrementalTreeState.from_layers(lk[:1], lv[:1], CR, MTN)
    with pytest.raises(ValueError, match="uniform"):
        IncrementalTreeState.from_layers(
            [lk[0], lk[1].double(), lk[2]], lv, CR, MTN)


def test_per_level_capacity_validated_at_init():
    k, v = _mk(60, seed=9)
    w = torch.randn(1, 8)
    tau = torch.zeros(1)
    # 60 tokens + 3000 reserve can need 3 pooled levels; 1 is not enough
    with pytest.raises(ValueError, match="capacity"):
        IncrementalTreeState(
            k, v, CR, MTN, reserve=3000, probe_w=w, probe_tau=tau,
            probe_mode="per_level",
        )
