"""Probe-attention pooling equivalence and gradient-path tests."""
from __future__ import annotations

import pytest
import torch

from recsa.backward import (
    RecSATritonFunction,
    recsa_forward_probe_autograd,
    recsa_forward_triton_autograd,
)
from recsa.backward import _recompute_tree_build
from recsa.probe_pool import build_tree_layers, compute_num_layers

pytestmark = pytest.mark.gpu

CR, TOPK = 16, 64
MTN = CR * TOPK
T, B, HQ, HKV, D = 16384, 1, 4, 2, 64


def _rand_qkv(dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, T, HQ, D, device="cuda", dtype=torch.float32, generator=g).to(dtype)
    k = torch.randn(B, T, HKV, D, device="cuda", dtype=torch.float32, generator=g).to(dtype)
    v = torch.randn(B, T, HKV, D, device="cuda", dtype=torch.float32, generator=g).to(dtype)
    return q, k, v


def test_tau0_probe_pyramid_matches_triton_mean_build(require_cuda):
    num_layers = compute_num_layers(T, CR, MTN)
    assert num_layers >= 2
    w = torch.randn(4, D, device="cuda", dtype=torch.float32) * D ** -0.5
    tau = torch.zeros(4, device="cuda", dtype=torch.float32)

    _, k, v = _rand_qkv()
    ref_k, ref_v = _recompute_tree_build(k, v, num_layers, CR)
    got_k, got_v = build_tree_layers(k, v, CR, MTN, probe_w=w, probe_tau=tau)
    for layer in range(num_layers):
        for name, got, ref in (("K", got_k[layer], ref_k[layer]), ("V", got_v[layer], ref_v[layer])):
            g, r = got.float(), ref.float()
            tol = 2.0 ** -7 * (r.abs() + 1.0)
            bad = ((g - r).abs() > tol).sum().item()
            assert bad == 0, f"{name} layer {layer}: {bad} elements beyond 1 bf16 ulp"

    # fp32 storage: semantic equality up to fp32 reduction-order noise.
    _, k32, v32 = _rand_qkv(dtype=torch.float32, seed=7)
    ref_k, ref_v = _recompute_tree_build(k32, v32, num_layers, CR)
    got_k, got_v = build_tree_layers(k32, v32, CR, MTN, probe_w=w, probe_tau=tau)
    for layer in range(num_layers):
        dk = (got_k[layer] - ref_k[layer]).abs().max().item()
        dv = (got_v[layer] - ref_v[layer]).abs().max().item()
        assert dk < 1e-5 and dv < 1e-5, f"fp32 layer {layer} diff k={dk} v={dv}"


def test_layers_path_matches_fused_path_fwd_bwd(require_cuda):
    q, k, v = _rand_qkv(dtype=torch.float32)
    scale = D ** -0.5

    q1 = q.clone().requires_grad_(True)
    k1 = k.clone().requires_grad_(True)
    v1 = v.clone().requires_grad_(True)
    out1 = recsa_forward_triton_autograd(q1, k1, v1, CR, MTN, TOPK, scale)
    out1.float().square().mean().backward()

    q2 = q.clone().requires_grad_(True)
    k2 = k.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)
    layers_k, layers_v = build_tree_layers(k2, v2, CR, MTN)  # mean pooling, differentiable
    from recsa.backward import RecSATritonLayersFunction
    out2 = RecSATritonLayersFunction.apply(
        q2, CR, MTN, TOPK, scale, 10000.0, "internal", len(layers_k), *layers_k, *layers_v,
    )
    out2.float().square().mean().backward()

    out_diff = (out1.float() - out2.float()).abs().max().item()
    assert out_diff <= 2e-2, f"output diff {out_diff}"
    for name, a, b in (("dq", q1.grad, q2.grad), ("dk", k1.grad, k2.grad), ("dv", v1.grad, v2.grad)):
        diff = (a.float() - b.float()).abs().max().item()
        ref = a.float().abs().max().item()
        assert diff <= max(2e-2 * ref, 2e-3), f"{name} diff {diff} vs ref magnitude {ref}"


def test_probe_gradients_flow_tau_first(require_cuda):
    """At tau=0 the probe path routes gradient to tau but not to w."""
    q, k, v = _rand_qkv(dtype=torch.float32, seed=3)
    w = (torch.randn(4, D, device="cuda") * D ** -0.5).requires_grad_(True)
    tau = torch.zeros(4, device="cuda", requires_grad=True)
    out = recsa_forward_probe_autograd(q, k, v, w, tau, CR, MTN, TOPK)
    out.float().square().mean().backward()

    num_pool_levels = compute_num_layers(T, CR, MTN) - 1
    assert torch.all(w.grad[:num_pool_levels].abs() < 1e-12), "w must have zero grad at tau=0"
    assert tau.grad[:num_pool_levels].abs().max().item() > 0, "tau must receive gradient at tau=0"
    assert torch.all(tau.grad[num_pool_levels:] == 0), "unused levels must stay untouched"


def test_pooling_graph_finite_difference(require_cuda):
    """Finite-difference check of the probe softmax pooling in isolation, where the loss is a smooth function
    of the pooled pyramid."""
    torch.manual_seed(11)
    k = torch.randn(1, 2048, 2, D, device="cuda", dtype=torch.float64)
    v = torch.randn(1, 2048, 2, D, device="cuda", dtype=torch.float64)
    probes = torch.randn(1, 128, 2, D, device="cuda", dtype=torch.float64)  # match level-1 node count

    def loss_fn(w_in, tau_in):
        lk, _ = build_tree_layers(k, v, 16, 128, probe_w=w_in, probe_tau=tau_in)
        return (lk[1].to(torch.float64) * probes).sum().square()

    w = (torch.randn(4, D, device="cuda", dtype=torch.float64) * D ** -0.5).requires_grad_(True)
    tau = torch.full((4,), 0.7, device="cuda", dtype=torch.float64, requires_grad=True)
    loss = loss_fn(w, tau)
    loss.backward()

    eps = 1e-5
    for name, param, grad in (("tau", tau, tau.grad), ("w", w, w.grad)):
        flat_idx = 0 if name == "tau" else 3  # tau[0]; w[0, 3]
        with torch.no_grad():
            shift = param.detach().clone()
            shift.view(-1)[flat_idx] += eps
            lp = loss_fn(shift if name == "w" else w.detach(), shift if name == "tau" else tau.detach()).item()
            shift2 = param.detach().clone()
            shift2.view(-1)[flat_idx] -= eps
            lm = loss_fn(shift2 if name == "w" else w.detach(), shift2 if name == "tau" else tau.detach()).item()
        fd = (lp - lm) / (2 * eps)
        an = grad.view(-1)[flat_idx].item()
        assert abs(fd - an) <= max(1e-4 * abs(fd), 1e-7), f"{name} grad analytic {an} vs FD {fd}"
