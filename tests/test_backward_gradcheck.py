from __future__ import annotations

import pytest
import torch

from recsa.reference.oracles import RecSAParams, fixed_selection_forward_from_parents, parents_from_naive_trace
from recsa.naive import recsa_forward_naive


@pytest.mark.slow
def test_fixed_selection_oracle_gradcheck_cpu():
    torch.manual_seed(505)
    dtype = torch.float64
    batch, seq_len, hq, h_kv, dim, v_dim = 1, 5, 1, 1, 4, 4
    params = RecSAParams(compression_rate=2, top_k_per_layer=2, max_top_nodes=4, scale=dim ** -0.5)
    q0 = torch.randn(batch, seq_len, hq, dim, dtype=dtype)
    k0 = torch.randn(batch, seq_len, h_kv, dim, dtype=dtype)
    v0 = torch.randn(batch, seq_len, h_kv, v_dim, dtype=dtype)

    with torch.no_grad():
        _, trace = recsa_forward_naive(
            q0,
            k0,
            v0,
            compression_rate=params.compression_rate,
            top_k_per_layer=params.top_k_per_layer,
            max_top_nodes=params.max_top_nodes,
            scale=params.scale,
            return_trace=True,
        )
    parents = parents_from_naive_trace(trace, batch, seq_len, h_kv, params, q0.device)

    def fn(q, k, v):
        # Selection is deliberately frozen outside the differentiable path.
        return fixed_selection_forward_from_parents(q, k, v, parents, params)

    q = q0.detach().clone().requires_grad_(True)
    k = k0.detach().clone().requires_grad_(True)
    v = v0.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(fn, (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-3)
