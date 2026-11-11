from __future__ import annotations

import pytest
import torch

from recsa.reference.oracles import dense_degenerate_reference
from recsa.testing.reporting import diff_stats
from recsa.parallel import recsa_forward


pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("hq,h_kv", [(2, 2), (4, 2)])
def test_forward_dense_degenerate_matches_dense_causal(require_cuda, record_m1_result, dtype_name, hq, h_kv):
    torch.manual_seed(100 + hq + h_kv)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    device = require_cuda
    batch, seq_len, dim, v_dim = 1, 17, 32, 32
    compression_rate = 16
    top_k_per_layer = 64
    max_top_nodes = compression_rate * top_k_per_layer
    q = torch.randn(batch, seq_len, hq, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, h_kv, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, h_kv, v_dim, device=device, dtype=dtype)

    out = recsa_forward(
        q,
        k,
        v,
        compression_rate=compression_rate,
        top_k_per_layer=top_k_per_layer,
        max_top_nodes=max_top_nodes,
    )
    ref = dense_degenerate_reference(q, k, v)
    stats = diff_stats(out, ref)
    record_m1_result({
        "check": "forward_dense_degenerate",
        "status": "pass" if (stats["p99_abs"] < (2e-3 if dtype == torch.float32 else 3e-2) and stats["max_abs"] < (5e-3 if dtype == torch.float32 else 8e-2)) else "fail",
        "dtype": dtype_name,
        "shape": {"B": batch, "T": seq_len, "Hq": hq, "Hkv": h_kv, "D": dim, "Dv": v_dim},
        **stats,
    })

    if dtype == torch.float32:
        assert stats["p99_abs"] < 2e-3 and stats["max_abs"] < 5e-3, stats
    else:
        assert stats["p99_abs"] < 3e-2 and stats["max_abs"] < 8e-2, stats
