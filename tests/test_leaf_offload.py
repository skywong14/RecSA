"""Leaf-offload tests."""
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
import recsa.incremental_tree as incremental_tree  # noqa: E402
import recsa.probe_pool as probe_pool  # noqa: E402
from recsa.decode import RecSADecodeSession  # noqa: E402
from recsa.incremental_tree import OpenParentVBatch, pinned_host_zeros  # noqa: E402
from recsa.probe_pool import BucketCursor, DecodeBucketPoolMLP  # noqa: E402
from recsa.parallel import (  # noqa: E402
    RECSA_GATHER_PROGS,
    recsa_leaf_gather_kernel,
    recsa_leaf_store_bucket_kernel,
    recsa_leaf_store_kernel,
)

pytestmark = pytest.mark.gpu

TK, MTN = 64, 1024


def _model():
    cfg = Qwen3Config(
        hidden_size=512, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=128, intermediate_size=1024,
        vocab_size=512, max_position_embeddings=65536, rope_theta=5e6,
        use_sliding_window=False, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def _graft(m, tmp_path=None, shared_probe=False):
    kw = dict(mode="internal", rope_base=5e6, top_k=TK, max_top_nodes=MTN)
    if shared_probe:
        torch.manual_seed(3)
        w = torch.randn(2, 128)
        w = w / w.norm(dim=-1, keepdim=True) * 12.35   # deployed ||w||
        ck = {"probe_w": w, "probe_tau": torch.full((2,), 0.7717),
              "probe_gate": torch.full((2,), 1.0), "probe_mode": "shared_std"}
        p = tmp_path / "probe_shared.pt"
        torch.save(ck, p)
        kw["probe_ckpt"] = str(p)
    install_recsa_attention(m, **kw)
    install_streamed_forward(m, q_chunk=4096)
    return m


def _assert_offloaded(sess):
    """The offload session keeps leaves on the host and upper levels on the GPU."""
    for st in sess.states:
        assert st._offload, "state did not enter offload mode"
        assert not st._k[0].is_cuda and st._k[0].is_pinned()
        assert st._k[1].is_cuda


def _run_pair(m, t0, n_steps, *, graph=False):
    """All-GPU eager reference vs offload session (eager or graphed) on the same prefix; per-step logits are equal."""
    torch.manual_seed(42)
    ids = torch.randint(0, 512, (1, t0 + n_steps), device="cuda")
    ref = RecSADecodeSession(m, reserve=n_steps + 8)
    ref.prefill(ids[:, :t0])
    got = RecSADecodeSession(m, reserve=n_steps + 8, offload_leaves=True)
    got.prefill(ids[:, :t0])
    _assert_offloaded(got)
    if graph:
        got.enable_cuda_graph()
    for i in range(n_steps):
        tok = ids[:, t0 + i:t0 + i + 1]
        a = ref.step(tok)
        b = got.step(tok)
        assert torch.equal(a, b), (
            f"step {i} (t={t0 + i}): offload logits diverge, "
            f"max|d|={(a - b).abs().max().item():g}")
    return ref, got


def _assert_trees_equal(ref, got):
    for s in (ref, got):
        for st in s.states:
            st.sync_sizes(s.t)
    for li, (a, b) in enumerate(zip(ref.states, got.states)):
        assert a._sizes == b._sizes, f"layer {li}: sizes {a._sizes} vs {b._sizes}"
        for lvl in range(len(a._k)):
            n = a._sizes[lvl]
            assert torch.equal(a._k[lvl][:, :n].cpu(), b._k[lvl][:, :n].cpu()), \
                f"layer {li} level {lvl} K"
            assert torch.equal(a._v[lvl][:, :n].cpu(), b._v[lvl][:, :n].cpu()), \
                f"layer {li} level {lvl} V"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("alloc", ["pin", "register"])
def test_store_gather_roundtrip(dtype, alloc):
    torch.manual_seed(7)
    H_kv, K, V, CR = 2, 128, 128, 16
    cap = 4096
    if alloc == "pin":
        pool_k = torch.zeros(1, cap, H_kv, K, dtype=dtype, pin_memory=True)
        pool_v = torch.zeros(1, cap, H_kv, V, dtype=dtype, pin_memory=True)
    else:
        pool_k = pinned_host_zeros((1, cap, H_kv, K), dtype)
        pool_v = pinned_host_zeros((1, cap, H_kv, V), dtype)
    assert pool_k.is_pinned() and pool_v.is_pinned()
    mirror_k = torch.zeros_like(pool_k)
    mirror_v = torch.zeros_like(pool_v)
    t_dev = torch.zeros(1, dtype=torch.int64, device="cuda")

    T = 700  # 43 full buckets + a 12-row tail
    for t in range(T):
        k_t = torch.randn(1, 1, H_kv, K, dtype=dtype, device="cuda")
        v_t = torch.randn(1, 1, H_kv, V, dtype=dtype, device="cuda")
        t_dev.fill_(t)
        recsa_leaf_store_kernel[(H_kv,)](
            pool_k, pool_v, k_t, v_t, t_dev, H_kv=H_kv, K=K, V=V)
        mirror_k[0, t] = k_t[0, 0].cpu()
        mirror_v[0, t] = v_t[0, 0].cpu()
    torch.cuda.synchronize()
    assert torch.equal(pool_k, mirror_k), "store kernel corrupted K bytes"
    assert torch.equal(pool_v, mirror_v), "store kernel corrupted V bytes"

    rm = (T - 1) // CR
    psp = torch.full((1, 1, H_kv, TK), -1, dtype=torch.int32, device="cuda")
    psp[0, 0, 0, :5] = torch.tensor([0, 3, rm, 17, rm], dtype=torch.int32)
    psp[0, 0, 1, :4] = torch.tensor([rm, 2, 2, 41], dtype=torch.int32)

    fetch_k = torch.zeros(H_kv, TK * CR, K, dtype=dtype, device="cuda")
    fetch_v = torch.zeros(H_kv, TK * CR, V, dtype=dtype, device="cuda")
    t_dev.fill_(T - 1)
    recsa_leaf_gather_kernel[(RECSA_GATHER_PROGS,)](
        pool_k, pool_v, fetch_k, fetch_v, psp, t_base=0,
        H_kv=H_kv, K=K, V=V, COMPRESSION_RATE=CR, TOP_K=TK,
        NPROG=RECSA_GATHER_PROGS, t_ptr=t_dev, HAS_CURSOR=1, num_warps=4)
    torch.cuda.synchronize()

    psp_c = psp.cpu()
    ar = torch.arange(CR)
    for h in range(H_kv):
        for p in range(TK):
            rows = (max(int(psp_c[0, 0, h, p]), 0) * CR + ar).clamp(0, T - 1)
            assert torch.equal(fetch_k[h, p * CR:(p + 1) * CR].cpu(),
                               mirror_k[0, rows, h]), f"K tile ({h},{p})"
            assert torch.equal(fetch_v[h, p * CR:(p + 1) * CR].cpu(),
                               mirror_v[0, rows, h]), f"V tile ({h},{p})"

    # same launch with a host t_base instead of the cursor
    fetch_k2 = torch.zeros_like(fetch_k)
    fetch_v2 = torch.zeros_like(fetch_v)
    recsa_leaf_gather_kernel[(RECSA_GATHER_PROGS,)](
        pool_k, pool_v, fetch_k2, fetch_v2, psp, t_base=T - 1,
        H_kv=H_kv, K=K, V=V, COMPRESSION_RATE=CR, TOP_K=TK,
        NPROG=RECSA_GATHER_PROGS, num_warps=4)
    torch.cuda.synchronize()
    assert torch.equal(fetch_k, fetch_k2) and torch.equal(fetch_v, fetch_v2), \
        "cursor vs host-arg gather twins diverge"


def test_offload_session_mean_eager_bitwise():
    m = _graft(_model())
    ref, got = _run_pair(m, 17140, 24, graph=False)
    _assert_trees_equal(ref, got)


def test_offload_session_probe_eager_bitwise(tmp_path):
    m = _graft(_model(), tmp_path, shared_probe=True)
    ref, got = _run_pair(m, 17140, 24, graph=False)
    _assert_trees_equal(ref, got)


def test_offload_session_mean_graph_bitwise():
    m = _graft(_model())
    ref, got = _run_pair(m, 17140, 24, graph=True)
    assert got._graph is not None, "graph never captured — gate is vacuous"
    _assert_trees_equal(ref, got)


def test_offload_session_probe_graph_bitwise(tmp_path):
    m = _graft(_model(), tmp_path, shared_probe=True)
    ref, got = _run_pair(m, 17140, 24, graph=True)
    assert got._graph is not None, "graph never captured — gate is vacuous"
    _assert_trees_equal(ref, got)


def test_offload_band_crossing_recaptures():
    m = _graft(_model())
    ref, got = _run_pair(m, 16370, 32, graph=True)
    assert got._graph is not None, "no recapture after the band crossing"
    assert got.states[0].num_layers == 3, "crossing never happened — vacuous"
    _assert_trees_equal(ref, got)


def test_registered_pool_lifecycle():
    import gc
    for _ in range(3):
        t = pinned_host_zeros((1, 2048, 2, 128), torch.bfloat16)
        assert t.is_pinned()
        del t
        gc.collect()


# -------------------------------------------------------------- canary ----

def test_offload_corruption_canary():
    """Flipping one pinned-pool row changes the next step's logits: the staged read path reads the pool."""
    m = _graft(_model())
    t0 = 17140
    torch.manual_seed(42)
    ids = torch.randint(0, 512, (1, t0 + 2), device="cuda")
    a = RecSADecodeSession(m, reserve=16, offload_leaves=True)
    a.prefill(ids[:, :t0])
    b = RecSADecodeSession(m, reserve=16, offload_leaves=True)
    b.prefill(ids[:, :t0])
    with torch.inference_mode():
        b.states[0]._v[0][0, t0 - 1, 0, :] = 1000.0
    tok = ids[:, t0:t0 + 1]
    la, lb = a.step(tok), b.step(tok)
    assert not torch.equal(la, lb), (
        "corrupted host pool left logits bitwise unchanged — the offload "
        "path is not reading the pool; every parity gate above is vacuous")


def _graft_mlp(m, tmp_path, mode, *, top_k=TK, mtn=MTN, h1=32, gate=1.0):
    D, H1 = 128, h1
    torch.manual_seed(3)
    w = torch.randn(2, D)
    w = w / w.norm(dim=-1, keepdim=True) * 12.35
    in_dim = D if mode == "mlp_scorer" else 2 * D
    w2_shape = (2, H1) if mode == "mlp_scorer" else (2, D, H1)
    torch.manual_seed(11)
    ck = {"probe_w": w, "probe_tau": torch.full((2,), 0.7717),
          "probe_gate": torch.full((2,), gate), "probe_mode": mode,
          "probe_extra": {
              "mlp_w1": torch.randn(2, H1, in_dim) * in_dim ** -0.5,
              "mlp_b1": torch.randn(2, H1) * 0.1,
              "mlp_w2": torch.randn(*w2_shape) * 0.05,
          }}
    p = tmp_path / f"probe_{mode}_tk{top_k}.pt"
    torch.save(ck, p)
    install_recsa_attention(m, mode="internal", rope_base=5e6, top_k=top_k,
                            max_top_nodes=mtn, probe_ckpt=str(p))
    install_streamed_forward(m, q_chunk=4096)
    return m


def _assert_mlp_live(sess, mode):
    """Every adopted state carries the MLP extras with a nonzero output layer."""
    for st in sess.states:
        assert st._mode == mode and st._probe_extra is not None
        assert float(st._probe_extra["mlp_w2"].abs().max()) > 0


@pytest.mark.parametrize("mode", ["mlp_scorer", "mlp_pooler"])
def test_offload_session_mlp_eager_bitwise(mode, tmp_path):
    m = _graft_mlp(_model(), tmp_path, mode)
    ref, got = _run_pair(m, 17140, 24, graph=False)
    _assert_mlp_live(ref, mode)
    _assert_mlp_live(got, mode)
    _assert_trees_equal(ref, got)


_INT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16,
             torch.float32: torch.int32}


def _bits(x):
    return x.view(_INT_VIEW[x.dtype])


@pytest.mark.parametrize("launch", ["direct", "graph"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("alloc", ["pin", "register"])
def test_store_bucket_roundtrip(dtype, alloc, launch):
    torch.manual_seed(7)
    H_kv, K, V, CR = 2, 128, 128, 16
    T = 700                       # 43 full buckets + a 12-row tail
    cap = T                       # the final buckets run past the pool end
    if alloc == "pin":
        pool_k = torch.empty(1, cap, H_kv, K, dtype=dtype, pin_memory=True)
        pool_v = torch.empty(1, cap, H_kv, V, dtype=dtype, pin_memory=True)
    else:
        pool_k = pinned_host_zeros((1, cap, H_kv, K), dtype)
        pool_v = pinned_host_zeros((1, cap, H_kv, V), dtype)
    assert pool_k.is_pinned() and pool_v.is_pinned()
    _bits(pool_k).fill_(-1)
    _bits(pool_v).fill_(-1)
    mirror_k, mirror_v = pool_k.clone(), pool_v.clone()
    stage_k = torch.empty(1, CR, H_kv, K, dtype=dtype, device="cuda")
    stage_v = torch.empty(1, CR, H_kv, V, dtype=dtype, device="cuda")
    k_in = torch.empty(1, 1, H_kv, K, dtype=dtype, device="cuda")
    v_in = torch.empty(1, 1, H_kv, V, dtype=dtype, device="cuda")
    t_dev = torch.zeros(1, dtype=torch.int64, device="cuda")

    def run():
        recsa_leaf_store_bucket_kernel[(H_kv,)](
            pool_k, pool_v, stage_k, stage_v, k_in, v_in, t_dev,
            H_kv=H_kv, K=K, V=V, C=CR)

    graph = None
    n_open = n_seal = n_hostrow_reads = n_past_end = n_launch = 0
    host_rows = set()
    for t in range(T):
        k_in.copy_(torch.randn(1, 1, H_kv, K, device="cuda").to(dtype))
        v_in.copy_(torch.randn(1, 1, H_kv, V, device="cuda").to(dtype))
        t_dev.fill_(t)
        mirror_k[0, t] = k_in[0, 0].cpu()
        mirror_v[0, t] = v_in[0, 0].cpu()
        if t % 37 == 5:
            # host path (eager append): a synchronous D2H row copy
            pool_k[:, t].copy_(k_in[:, 0])
            pool_v[:, t].copy_(v_in[:, 0])
            host_rows.add(t)
            continue
        _bits(stage_k).fill_(-1)
        _bits(stage_v).fill_(-1)
        if launch == "graph" and graph is not None:
            graph.replay()
        else:
            run()
            if launch == "graph":
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()            # recorded, not executed
        torch.cuda.synchronize()
        n_launch += 1
        lo, n = t // CR * CR, t % CR + 1
        want_k = torch.zeros(1, CR, H_kv, K, dtype=dtype)
        want_v = torch.zeros(1, CR, H_kv, V, dtype=dtype)
        want_k[:, :n] = mirror_k[:, lo:t + 1]
        want_v[:, :n] = mirror_v[:, lo:t + 1]
        assert torch.equal(_bits(stage_k.cpu()), _bits(want_k)), f"t={t}: staged K bucket"
        assert torch.equal(_bits(stage_v.cpu()), _bits(want_v)), f"t={t}: staged V bucket"
        n_open += n == 1
        n_seal += n == CR
        n_hostrow_reads += any(r in host_rows for r in range(lo, t))
        n_past_end += lo + CR > cap
    assert torch.equal(_bits(pool_k), _bits(mirror_k)), "pool K bytes"
    assert torch.equal(_bits(pool_v), _bits(mirror_v)), "pool V bytes"
    # sweep coverage
    assert n_launch > 600 and n_open >= 40 and n_seal >= 40
    assert n_hostrow_reads >= 100, "host-path rows never read back"
    assert n_past_end >= 5, "no bucket ran past the pool end"


def _run_offload_graph_triple(m, t0, n_steps):
    torch.manual_seed(42)
    ids = torch.randint(0, 512, (1, t0 + n_steps), device="cuda")
    ref = RecSADecodeSession(m, reserve=n_steps + 8)
    ref.prefill(ids[:, :t0])
    off_e = RecSADecodeSession(m, reserve=n_steps + 8, offload_leaves=True)
    off_e.prefill(ids[:, :t0])
    off_g = RecSADecodeSession(m, reserve=n_steps + 8, offload_leaves=True)
    off_g.prefill(ids[:, :t0])
    _assert_offloaded(off_e)
    _assert_offloaded(off_g)
    off_g.enable_cuda_graph()
    for st in off_g.states:
        assert st._stage_k is not None and st._stage_v is not None, \
            "staging pair not bound — the MLP offload graph path is not taken"
    for i in range(n_steps):
        tok = ids[:, t0 + i:t0 + i + 1]
        a = ref.step(tok)
        b = off_e.step(tok)
        c = off_g.step(tok)
        assert torch.equal(a, b), (
            f"step {i} (t={t0 + i}): offload EAGER logits diverge, "
            f"max|d|={(a - b).abs().max().item():g}")
        assert torch.equal(a, c), (
            f"step {i} (t={t0 + i}): offload GRAPH logits diverge, "
            f"max|d|={(a - c).abs().max().item():g}")
    return ref, off_e, off_g


def _assert_stage_holds_last_bucket(sess):
    """The last step (a graph replay) refilled every layer's staging pair from the pool: rows lo..t are the pool's, the
    pad rows are +0.0."""
    t = sess.t - 1
    for li, st in enumerate(sess.states):
        lo, n = t // st.c * st.c, t % st.c + 1
        for stage, pool, kv in ((st._stage_k, st._k[0], "K"), (st._stage_v, st._v[0], "V")):
            s = stage.cpu()
            assert torch.equal(_bits(s[:, :n]), _bits(pool[:, lo:t + 1])), \
                f"layer {li}: staged {kv} rows != pool rows {lo}..{t}"
            assert not _bits(s[:, n:]).any(), f"layer {li}: staged {kv} pad rows not +0.0"


@pytest.mark.parametrize("mode", [
    "mlp_pooler",
    pytest.param("mlp_scorer", marks=pytest.mark.full),   # not the default probe mode
])
def test_offload_mlp_graph_bitwise(mode, tmp_path):
    m = _graft_mlp(_model(), tmp_path, mode)
    ref, off_e, off_g = _run_offload_graph_triple(m, 17140, 24)
    assert off_g._graph is not None, "graph never captured — gate is vacuous"
    for s in (ref, off_e, off_g):
        _assert_mlp_live(s, mode)
    _assert_stage_holds_last_bucket(off_g)
    _assert_trees_equal(ref, off_g)
    _assert_trees_equal(ref, off_e)


def test_offload_mlp_graph_bitwise_tk256_nl3(tmp_path):
    m = _graft_mlp(_model(), tmp_path, "mlp_pooler", top_k=256, mtn=4096,
                   h1=64, gate=0.85)
    ref, off_e, off_g = _run_offload_graph_triple(m, 65774, 24)
    assert off_g._graph is not None, "graph never captured — gate is vacuous"
    assert off_g.states[0].num_layers == 3, "not at NL=3 — gate is vacuous"
    _assert_mlp_live(off_g, "mlp_pooler")
    _assert_stage_holds_last_bucket(off_g)
    _assert_trees_equal(ref, off_g)
    _assert_trees_equal(ref, off_e)


def test_offload_mlp_graph_band_crossing_tk256(tmp_path):
    m = _graft_mlp(_model(), tmp_path, "mlp_pooler", top_k=256, mtn=4096,
                   h1=64, gate=0.85)
    ref, off_e, off_g = _run_offload_graph_triple(m, 65524, 32)
    assert off_g._graph is not None, "no recapture after the band crossing"
    assert len(getattr(off_g, "_g_retired", [])) == 1, \
        "no graph before the crossing — vacuous"
    assert off_g.states[0].num_layers == 3, "crossing never happened — vacuous"
    _assert_stage_holds_last_bucket(off_g)
    _assert_trees_equal(ref, off_g)
    _assert_trees_equal(ref, off_e)


def test_offload_mlp_graph_flag_off_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(incremental_tree, "MLP_OFFLOAD_GRAPH", "off")
    m = _graft_mlp(_model(), tmp_path, "mlp_pooler")
    torch.manual_seed(42)
    ids = torch.randint(0, 512, (1, 4096), device="cuda")
    s = RecSADecodeSession(m, reserve=8, offload_leaves=True)
    s.prefill(ids)
    with pytest.raises(ValueError, match="fused pool.*RECSA_MLP_OFFLOAD_GRAPH=off"):
        s.enable_cuda_graph()
    assert not getattr(s, "_graph_on", False)


def test_offload_mlp_graph_reads_pool_canary(tmp_path):
    m = _graft_mlp(_model(), tmp_path, "mlp_pooler")
    t0, warm = 17140, 4           # 2 cursor-eager warm steps, capture, 1 replay
    torch.manual_seed(42)
    ids = torch.randint(0, 512, (1, t0 + warm + 1), device="cuda")
    sess = {}
    for name in ("e", "g", "h"):  # eager / graphed / graphed, never corrupted
        s = RecSADecodeSession(m, reserve=16, offload_leaves=True)
        s.prefill(ids[:, :t0])
        if name != "e":
            s.enable_cuda_graph()
        sess[name] = s
    for i in range(warm):
        tok = ids[:, t0 + i:t0 + i + 1]
        le, lg, lh = (sess[n].step(tok) for n in ("e", "g", "h"))
        assert torch.equal(le, lg) and torch.equal(le, lh), f"step {i}"
    assert sess["g"]._graph is not None and sess["h"]._graph is not None
    t = t0 + warm                 # the next append's row
    c = sess["e"].states[0].c
    assert t % c != 0, "the corrupted row must share the open bucket"
    torch.cuda.synchronize()
    with torch.inference_mode():
        for n in ("e", "g"):
            sess[n].states[0]._k[0][0, t - 1, 0, :] = 4.0
    tok = ids[:, t:t + 1]
    le, lg, lh = (sess[n].step(tok) for n in ("e", "g", "h"))
    p = t // c
    pk = {n: sess[n].states[0]._k[1][0, p].clone() for n in sess}
    assert torch.equal(pk["e"], pk["g"]), \
        "graphed level-1 parent != eager on the corrupted pool"
    assert not torch.equal(pk["g"], pk["h"]), \
        "corrupted pool row did not reach the graphed level-1 pool — vacuous"
    assert torch.equal(le, lg), "graphed logits != eager on the corrupted pool"
    assert not torch.equal(lg, lh), "corruption invisible in the logits — vacuous"


def test_offload_mlp_graph_keeps_a7_path_under_pool_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_pool, "RECSA_MLP_CURSOR_POOL", "v2")
    monkeypatch.setattr(probe_pool, "RECSA_MLP_POOL_SEGMENTS", "exact")
    m = _graft_mlp(_model(), tmp_path, "mlp_pooler")
    t0, n = 17140, 6                  # 2 cursor-eager warm steps, capture, replays
    torch.manual_seed(42)
    ids = torch.randint(0, 512, (1, t0 + n), device="cuda")
    res = RecSADecodeSession(m, reserve=n + 8)
    res.prefill(ids[:, :t0])
    off = RecSADecodeSession(m, reserve=n + 8, offload_leaves=True)
    off.prefill(ids[:, :t0])
    _assert_offloaded(off)
    calls = {"cursor": 0, "cursor_k": 0, "cursor_k_exact": 0, "vbatch": 0}
    orig = {"cursor": DecodeBucketPoolMLP.cursor, "cursor_k": DecodeBucketPoolMLP.cursor_k,
            "cursor_k_exact": DecodeBucketPoolMLP.cursor_k_exact,
            "vbatch": OpenParentVBatch.__call__}

    def tap(name):
        def g(*a, **kw):
            calls[name] += 1
            return orig[name](*a, **kw)
        return g
    try:
        for name in ("cursor", "cursor_k", "cursor_k_exact"):
            setattr(DecodeBucketPoolMLP, name, tap(name))
        OpenParentVBatch.__call__ = tap("vbatch")
        res.enable_cuda_graph()
        off.enable_cuda_graph()
        assert all(st._defer_v for st in res.states) and res._g_vb is not None \
            and res._g_win.fused, "resident session not on the deferred-V cursor path — vacuous"
        assert not any(st._defer_v or st._exact for st in off.states)
        assert off._g_vb is None and not off._g_win.fused
        assert all(st._stage_k is not None for st in off.states), "staging pair not bound"
        for i in range(n):
            tok = ids[:, t0 + i:t0 + i + 1]
            before = dict(calls)
            b = off.step(tok)
            assert calls["cursor_k"] == before["cursor_k"] \
                and calls["cursor_k_exact"] == before["cursor_k_exact"] \
                and calls["vbatch"] == before["vbatch"], f"step {i}: offload took the resident cursor pool"
            a = res.step(tok)
            assert torch.equal(a, b), (
                f"step {i} (t={t0 + i}): offload graph != resident graph, "
                f"max|d|={(a - b).abs().max().item():g}")
    finally:
        for name in ("cursor", "cursor_k", "cursor_k_exact"):
            setattr(DecodeBucketPoolMLP, name, orig[name])
        OpenParentVBatch.__call__ = orig["vbatch"]
    assert off._graph is not None and res._graph is not None, "graph never captured — vacuous"
    # python-visible steps: 2 warm + 1 capture per session, every layer x pooled level
    per = 3 * len(off.states) * (off.states[0].num_layers - 1)
    assert calls == {"cursor": per, "cursor_k": per, "cursor_k_exact": 0, "vbatch": 3}, calls
    _assert_mlp_live(off, "mlp_pooler")
    _assert_trees_equal(res, off)
    st = off.states[0]                # the session is finished: probe the guard
    win = BucketCursor(off._t_dev, st.c, st._k[0].shape[3], st._capacity, fused=True)
    with pytest.raises(ValueError, match="needs resident leaves"):
        st.bind_cursor(off._t_dev, win, defer_v=True)
