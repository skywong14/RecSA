"""Per-node selection bias b from the dispersion of the pooling weights."""
from __future__ import annotations

import torch


def bucket_alpha(child_k, n_valid, w, tau, c):
    """Pooling weights alpha of the shared_std softmax, as in build_tree_layers. child_k [N,H,D] -> alpha [P,c,H]."""
    n_child, H, D = child_k.shape
    n_parent = (n_child + c - 1) // c
    pad = n_parent * c - n_child
    kf = child_k
    if pad:
        kf = torch.nn.functional.pad(kf, (0, 0, 0, 0, 0, pad))
    kf = kf.reshape(n_parent, c, H, D)
    valid = (torch.arange(n_parent * c, device=kf.device)
             .reshape(n_parent, c) < n_valid)
    count = valid.sum(1).clamp(min=1)
    mu = kf.sum(1, keepdim=True) / count[:, None, None, None]
    dev = (kf - mu).masked_fill(~valid[:, :, None, None], 0.0)
    var = dev.pow(2).sum(dim=(1, 3)) / (count[:, None] * D)
    sigma = var.clamp_min(1e-12).sqrt()
    logits = tau * (dev @ w) / sigma[:, None, :]
    logits = logits.masked_fill(~valid[:, :, None], float("-inf"))
    return torch.softmax(logits, dim=1)


def entropy(alpha):
    return -(alpha * alpha.clamp_min(1e-12).log()).sum(1)   # [P,H]


@torch.no_grad()
def compose_level_bias(layer_ks, n_leaves, w, tau, g_b, c):
    """Per-level bias tensors for recsa_forward(level_bias=...)."""
    g = min(max(float(g_b), 0.0), 1.0)
    if g == 0.0:
        return None
    assert layer_ks[0].shape[0] == 1, "eval path is B==1"
    out = [None]
    h_prev = None
    n_valid = int(n_leaves)
    for lvl in range(1, len(layer_ks)):
        child_k = layer_ks[lvl - 1][0].float()
        alpha = bucket_alpha(child_k, n_valid, w, tau, c)     # [P,c,H]
        h_here = entropy(alpha)                               # [P,H]
        if h_prev is not None:
            n_parent = alpha.shape[0]
            pad = n_parent * c - h_prev.shape[0]
            hp = torch.nn.functional.pad(h_prev, (0, 0, 0, pad))
            h_here = h_here + (alpha * hp.reshape(n_parent, c, -1)).sum(1)
        out.append((g * h_here)[None].contiguous().float())   # [1,P,H]
        h_prev = h_here
        n_valid = (n_valid + c - 1) // c
    return out
