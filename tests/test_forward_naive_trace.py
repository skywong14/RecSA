from __future__ import annotations

import pytest
import torch

from tests.correctness_test import run_parallel_with_trace
from recsa.testing.reporting import diff_stats
from recsa.naive import recsa_forward_naive


pytestmark = pytest.mark.gpu


def _parent_mismatch_count(trace_a, trace_b) -> int:
    total = 0
    for layer_idx, layer_a in trace_a["layers"].items():
        if layer_idx == 0 or layer_a.selected_parents is None:
            continue
        layer_b = trace_b["layers"][layer_idx]
        a = layer_a.selected_parents
        b = layer_b.selected_parents
        valid_shape = a.shape[:-1]
        for idx in torch.cartesian_prod(*[torch.arange(s, device=a.device) for s in valid_shape]):
            ai = a[tuple(idx.tolist())]
            bi = b[tuple(idx.tolist())]
            aset = set(int(x) for x in ai.detach().cpu().tolist() if x >= 0)
            bset = set(int(x) for x in bi.detach().cpu().tolist() if x >= 0)
            if aset != bset:
                total += 1
    return total


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_forward_naive_output_and_selected_parent_trace(require_cuda, record_m1_result, dtype_name):
    torch.manual_seed(2026)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    device = require_cuda
    batch, seq_len, hq, h_kv, dim, v_dim = 1, 1025, 4, 2, 32, 32
    compression_rate = 16
    top_k_per_layer = 64
    max_top_nodes = compression_rate * top_k_per_layer
    q = torch.randn(batch, seq_len, hq, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, h_kv, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, h_kv, v_dim, device=device, dtype=dtype)
    kwargs = dict(
        compression_rate=compression_rate,
        top_k_per_layer=top_k_per_layer,
        max_top_nodes=max_top_nodes,
        scale=dim ** -0.5,
    )

    with torch.no_grad():
        out_tri, trace_tri = run_parallel_with_trace(q, k, v, rope_base=10000.0, **kwargs)
        out_ref, trace_ref = recsa_forward_naive(q, k, v, return_trace=True, rope_base=10000.0, **kwargs)

    stats = diff_stats(out_tri, out_ref)
    mismatches = _parent_mismatch_count(trace_tri, trace_ref)
    status = "pass" if (stats["p99_abs"] < 3e-2 and mismatches == 0) else "fail"
    record_m1_result({
        "check": "forward_naive_trace",
        "status": status,
        "dtype": dtype_name,
        "shape": {"B": batch, "T": seq_len, "Hq": hq, "Hkv": h_kv, "D": dim, "Dv": v_dim},
        "parent_mismatch_count": mismatches,
        **stats,
    })
    assert stats["p99_abs"] < 3e-2, stats
    assert mismatches == 0
