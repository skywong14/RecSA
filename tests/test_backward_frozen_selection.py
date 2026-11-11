from __future__ import annotations

import pytest
import torch

from recsa.reference.oracles import RecSAParams, fixed_selection_forward_from_parents
from recsa.testing.reporting import diff_stats
from recsa.backward import recsa_backward
from recsa.parallel import recsa_forward


pytestmark = pytest.mark.gpu


def test_backward_matches_frozen_selection_autograd_fp32(require_cuda, record_m1_result):
    torch.manual_seed(404)
    device = require_cuda
    dtype = torch.float32
    batch, seq_len, hq, h_kv, dim, v_dim = 1, 1025, 2, 1, 32, 32
    params = RecSAParams(compression_rate=16, top_k_per_layer=64, max_top_nodes=1024, scale=dim ** -0.5)

    q0 = torch.randn(batch, seq_len, hq, dim, device=device, dtype=dtype)
    k0 = torch.randn(batch, seq_len, h_kv, dim, device=device, dtype=dtype)
    v0 = torch.randn(batch, seq_len, h_kv, v_dim, device=device, dtype=dtype)
    query_indices = torch.tensor([0, 15, 16, seq_len - 1], device=device, dtype=torch.long)
    grad_out_subset = torch.randn(batch, query_indices.numel(), hq, v_dim, device=device, dtype=dtype)
    grad_out = torch.zeros(batch, seq_len, hq, v_dim, device=device, dtype=dtype)
    grad_out[:, query_indices] = grad_out_subset

    with torch.no_grad():
        _, bwd_ctx = recsa_forward(
            q0,
            k0,
            v0,
            compression_rate=params.compression_rate,
            top_k_per_layer=params.top_k_per_layer,
            max_top_nodes=params.max_top_nodes,
            scale=params.scale,
            _save_for_backward=True,
            _save_parents=True,
        )
        parents = {layer: value.detach().clone() for layer, value in bwd_ctx["per_layer_parents"].items()}

    q_ref = q0.detach().clone().requires_grad_(True)
    k_ref = k0.detach().clone().requires_grad_(True)
    v_ref = v0.detach().clone().requires_grad_(True)
    out_ref = fixed_selection_forward_from_parents(q_ref, k_ref, v_ref, parents, params, query_indices=query_indices)
    out_ref.backward(grad_out_subset)

    q_tri = q0.detach().clone()
    k_tri = k0.detach().clone()
    v_tri = v0.detach().clone()
    out_tri, tri_ctx = recsa_forward(
        q_tri,
        k_tri,
        v_tri,
        compression_rate=params.compression_rate,
        top_k_per_layer=params.top_k_per_layer,
        max_top_nodes=params.max_top_nodes,
        scale=params.scale,
        _save_for_backward=True,
        _save_parents=True,
    )
    dq_tri, dk_tri, dv_tri = recsa_backward(grad_out, q_tri, out_tri, tri_ctx)

    records = {
        "out": diff_stats(out_tri[:, query_indices], out_ref),
        "dq": diff_stats(dq_tri, q_ref.grad),
        "dk": diff_stats(dk_tri, k_ref.grad),
        "dv": diff_stats(dv_tri, v_ref.grad),
    }
    status = "pass" if all(stats["p99_abs"] < 1e-2 for stats in records.values()) else "fail"
    record_m1_result({
        "check": "backward_frozen_selection",
        "status": status,
        "dtype": "float32",
        "shape": {"B": batch, "T": seq_len, "Hq": hq, "Hkv": h_kv, "D": dim, "Dv": v_dim},
        "query_indices": query_indices.detach().cpu().tolist(),
        "stats": records,
    })
    for name, stats in records.items():
        assert stats["p99_abs"] < 1e-2, {name: stats}
