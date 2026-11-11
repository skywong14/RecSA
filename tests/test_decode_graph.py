"""CUDA-graph decode tests."""
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

TK, MTN = 64, 1024

FULL = pytest.mark.full


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


def _run_pair(m, t0, n_steps):
    """Two sessions off the same prefix: eager reference vs graphed."""
    torch.manual_seed(42)
    ids = torch.randint(0, 512, (1, t0 + n_steps), device="cuda")
    ref = RecSADecodeSession(m, reserve=n_steps + 8)
    ref.prefill(ids[:, :t0])
    got = RecSADecodeSession(m, reserve=n_steps + 8)
    got.prefill(ids[:, :t0])
    got.enable_cuda_graph()
    for i in range(n_steps):
        tok = ids[:, t0 + i:t0 + i + 1]
        a = ref.step(tok)
        b = got.step(tok)
        assert torch.equal(a, b), (
            f"step {i} (t={t0 + i}): graphed logits diverge, "
            f"max|d|={(a - b).abs().max().item():g}")
    return ref, got


def _assert_trees_equal(ref, got):
    for st in got.states:
        st.sync_sizes(got.t)
    for li, (a, b) in enumerate(zip(ref.states, got.states)):
        assert a._sizes == b._sizes, f"layer {li}: sizes {a._sizes} vs {b._sizes}"
        for lvl, (ka, kb) in enumerate(zip(a._k, b._k)):
            n = a._sizes[lvl]
            assert torch.equal(ka[:, :n], kb[:, :n]), f"layer {li} level {lvl} K"
        for lvl, (va, vb) in enumerate(zip(a._v, b._v)):
            n = a._sizes[lvl]
            assert torch.equal(va[:, :n], vb[:, :n]), f"layer {li} level {lvl} V"


@FULL
def test_graph_session_mean_bitwise():
    m = _graft(_model())
    ref, got = _run_pair(m, 17140, 24)
    assert got._graph is not None, "graph never captured — gate is vacuous"
    _assert_trees_equal(ref, got)


@FULL
def test_graph_session_probe_bitwise(tmp_path):
    m = _graft(_model(), tmp_path, shared_probe=True)
    ref, got = _run_pair(m, 17140, 24)
    assert got._graph is not None, "graph never captured — gate is vacuous"
    _assert_trees_equal(ref, got)


@FULL
def test_graph_band_crossing_recaptures():
    m = _graft(_model())
    ref, got = _run_pair(m, 16370, 32)
    assert got._graph is not None, "no recapture after the band crossing"
    assert got.states[0].num_layers == 3, "crossing never happened — vacuous"
    _assert_trees_equal(ref, got)


def _graft512(m, tmp_path=None, shared_probe=False, top_k=512,
              max_top_nodes=8192):
    kw = dict(mode="internal", rope_base=5e6, top_k=top_k,
              max_top_nodes=max_top_nodes)
    if shared_probe:
        torch.manual_seed(3)
        w = torch.randn(2, 128)
        w = w / w.norm(dim=-1, keepdim=True) * 12.35   # deployed ||w||
        ck = {"probe_w": w, "probe_tau": torch.full((2,), 0.7717),
              "probe_gate": torch.full((2,), 1.0), "probe_mode": "shared_std"}
        p = tmp_path / f"probe_shared{top_k}.pt"
        torch.save(ck, p)
        kw["probe_ckpt"] = str(p)
    install_recsa_attention(m, **kw)
    install_streamed_forward(m, q_chunk=4096)
    return m


@pytest.mark.parametrize("top_k,mtn,t0", [
    pytest.param(512, 8192, 131310, id="tk512", marks=FULL),
    pytest.param(256, 4096, 65774, id="tk256"),
])
def test_graph_session_probe_bitwise_tk512_nl3(tmp_path, top_k, mtn, t0):
    m = _graft512(_model(), tmp_path, shared_probe=True, top_k=top_k,
                  max_top_nodes=mtn)
    ref, got = _run_pair(m, t0, 24)
    assert got._graph is not None, "graph never captured — gate is vacuous"
    assert got.states[0].num_layers == 3, "not at NL=3 — gate is vacuous"
    _assert_trees_equal(ref, got)


@pytest.mark.parametrize("top_k,mtn,t0", [
    pytest.param(512, 8192, 131060, id="tk512", marks=FULL),  # crossing at t+1=131,073
    pytest.param(256, 4096, 65524, id="tk256"),    # crossing at t+1= 65,537
])
def test_graph_band_crossing_recaptures_tk512(top_k, mtn, t0):
    m = _graft512(_model(), top_k=top_k, max_top_nodes=mtn)
    ref, got = _run_pair(m, t0, 32)
    assert got._graph is not None, "no recapture after the band crossing"
    assert got.states[0].num_layers == 3, "crossing never happened — vacuous"
    _assert_trees_equal(ref, got)


def _graft_mlp256(m, tmp_path, mode):
    D, H1 = 128, 64
    torch.manual_seed(3)
    w = torch.randn(2, D)
    w = w / w.norm(dim=-1, keepdim=True) * 12.35   # deployed ||w||
    in_dim = D if mode == "mlp_scorer" else 2 * D
    w2_shape = (2, H1) if mode == "mlp_scorer" else (2, D, H1)
    torch.manual_seed(11)
    ck = {"probe_w": w, "probe_tau": torch.full((2,), 0.7717),
          "probe_gate": torch.full((2,), 0.85), "probe_mode": mode,
          "probe_extra": {
              "mlp_w1": torch.randn(2, H1, in_dim) * in_dim ** -0.5,
              "mlp_b1": torch.randn(2, H1) * 0.1,
              "mlp_w2": torch.randn(*w2_shape) * 0.05,
          }}
    p = tmp_path / f"probe_{mode}256.pt"
    torch.save(ck, p)
    install_recsa_attention(m, mode="internal", rope_base=5e6, top_k=256,
                            max_top_nodes=4096, probe_ckpt=str(p))
    install_streamed_forward(m, q_chunk=4096)
    return m


def _assert_mlp_graphed(sess, mode):
    assert sess._g_win is not None, "no BucketCursor — MLP cursor path not taken"
    for st in sess.states:
        assert st._mode == mode and st._cursor_win is sess._g_win
        assert float(st._probe_extra["mlp_w2"].abs().max()) > 0


@pytest.mark.parametrize("mode", [
    "mlp_pooler",
    pytest.param("mlp_scorer", marks=FULL),   # not the default probe mode
])
def test_graph_session_mlp_bitwise_tk256_nl3(tmp_path, mode):
    m = _graft_mlp256(_model(), tmp_path, mode)
    ref, got = _run_pair(m, 65774, 24)
    assert got._graph is not None, "graph never captured — gate is vacuous"
    assert got.states[0].num_layers == 3, "not at NL=3 — gate is vacuous"
    _assert_mlp_graphed(got, mode)
    _assert_trees_equal(ref, got)


def test_graph_band_crossing_recaptures_mlp_pooler_tk256(tmp_path):
    m = _graft_mlp256(_model(), tmp_path, "mlp_pooler")
    ref, got = _run_pair(m, 65524, 32)
    assert got._graph is not None, "no recapture after the band crossing"
    assert len(getattr(got, "_g_retired", [])) == 1, \
        "no graph before the crossing — vacuous"
    assert got.states[0].num_layers == 3, "crossing never happened — vacuous"
    _assert_mlp_graphed(got, "mlp_pooler")
    _assert_trees_equal(ref, got)
