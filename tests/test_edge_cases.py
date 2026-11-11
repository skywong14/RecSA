from __future__ import annotations

import pytest
import torch

from recsa.backward import recsa_forward_triton_autograd
from recsa.parallel import recsa_forward


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 1023, 1024, 1025, 16383])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_forward_backward_edge_case_matrix(require_cuda, record_m1_result, seq_len, dtype_name):
    torch.manual_seed(808 + seq_len)
    device = require_cuda
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    batch, hq, h_kv, dim, v_dim = 2, 4, 2, 32, 32
    compression_rate = 16
    top_k_per_layer = 64
    max_top_nodes = 1024
    q = torch.randn(batch, seq_len, hq, dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, seq_len, h_kv, dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, seq_len, h_kv, v_dim, device=device, dtype=dtype, requires_grad=True)

    out = recsa_forward_triton_autograd(
        q,
        k,
        v,
        compression_rate=compression_rate,
        top_k_per_layer=top_k_per_layer,
        max_top_nodes=max_top_nodes,
        scale=dim ** -0.5,
    )
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    finite_ok = all(torch.isfinite(x).all().item() for x in [out, q.grad, k.grad, v.grad])
    record_m1_result({
        "check": "edge_case_matrix",
        "status": "pass" if finite_ok else "fail",
        "dtype": dtype_name,
        "shape": {"B": batch, "T": seq_len, "Hq": hq, "Hkv": h_kv, "D": dim, "Dv": v_dim},
        "finite_ok": bool(finite_ok),
    })
    assert finite_ok


def test_unsupported_gqa_configuration_fails_clearly(require_cuda):
    device = require_cuda
    q = torch.randn(1, 8, 3, 32, device=device)
    k = torch.randn(1, 8, 2, 32, device=device)
    v = torch.randn(1, 8, 2, 32, device=device)
    with pytest.raises(AssertionError, match="divisible"):
        recsa_forward(q, k, v, compression_rate=16, top_k_per_layer=64, max_top_nodes=1024)
