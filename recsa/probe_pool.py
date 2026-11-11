"""Differentiable tree building with probe-attention pooling."""
from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

RECSA_MLP_CURSOR_POOL = os.environ.get("RECSA_MLP_CURSOR_POOL", "v2").strip().lower()
if RECSA_MLP_CURSOR_POOL not in ("v2", "legacy"):
    raise ValueError("RECSA_MLP_CURSOR_POOL must be 'v2' or 'legacy', "
                     f"got {RECSA_MLP_CURSOR_POOL!r}")
RECSA_MLP_POOL_SEGMENTS = os.environ.get("RECSA_MLP_POOL_SEGMENTS", "fused").strip().lower()
if RECSA_MLP_POOL_SEGMENTS not in ("exact", "torch", "fused"):
    raise ValueError("RECSA_MLP_POOL_SEGMENTS must be 'fused', 'exact' or 'torch', "
                     f"got {RECSA_MLP_POOL_SEGMENTS!r}")

MLP_MODES = frozenset({"mlp_scorer", "mlp_pooler"})


def unpack_probe(probe):
    """Normalize an attn._recsa_probe tuple of len 2/3/4 to (w, tau, gate, extra)."""
    if probe is None:
        return None, None, None, None
    parts = tuple(probe) + (None,) * (4 - len(probe))
    return parts[0], parts[1], parts[2], parts[3]


def validate_probe_extra(probe_mode: str, probe_extra: Optional[Dict[str, torch.Tensor]],
                         k_dim: int) -> None:
    """Shape-check one layer's extra params before the build."""
    if probe_mode not in MLP_MODES:
        if probe_extra:
            raise ValueError(f"probe_extra given but mode is {probe_mode!r}")
        return
    if not probe_extra:
        raise ValueError(f"{probe_mode} requires probe_extra (mlp_w1/mlp_b1/mlp_w2)")
    w1, b1, w2 = probe_extra["mlp_w1"], probe_extra["mlp_b1"], probe_extra["mlp_w2"]
    h1 = w1.shape[0]
    in_dim = k_dim if probe_mode == "mlp_scorer" else 2 * k_dim
    if w1.shape != (h1, in_dim) or b1.shape != (h1,):
        raise ValueError(f"{probe_mode}: mlp_w1 {tuple(w1.shape)} / mlp_b1 "
                         f"{tuple(b1.shape)} inconsistent (expect [H1,{in_dim}]/[H1])")
    want_w2 = (h1,) if probe_mode == "mlp_scorer" else (k_dim, h1)
    if tuple(w2.shape) != want_w2:
        raise ValueError(f"{probe_mode}: mlp_w2 {tuple(w2.shape)} != {want_w2}")


def compute_num_layers(seq_len: int, compression_rate: int, max_top_nodes: int) -> int:
    num_layers = 1
    temp_len = seq_len
    while temp_len > max_top_nodes:
        temp_len = (temp_len + compression_rate - 1) // compression_rate
        num_layers += 1
    return num_layers


def _fused_pool_ok(child_k, child_v, c, probe_w, probe_tau, probe_gate, n_valid) -> bool:
    """recsa/pool_fused.py eligibility (prefill shapes on a supported torch / Triton build, no autograd)."""
    if probe_w is None or probe_tau is None or not child_k.is_cuda:
        return False
    from recsa import pool_fused
    return pool_fused.fused_ok(child_k, child_v, c, probe_w, probe_tau, probe_gate, n_valid)


def _pool_one_level(
    child_k: torch.Tensor,  # [B, N_child, H_kv, K]
    child_v: torch.Tensor,  # [B, N_child, H_kv, V]
    compression_rate: int,
    probe_w: Optional[torch.Tensor],  # [K] fp32 or None (mean)
    probe_tau: Optional[torch.Tensor],  # [] fp32 or None (mean)
    *,
    standardize: bool = False,          # score on (k - mu)/sigma per bucket
    probe_gate: Optional[torch.Tensor] = None,  # [] fp32; k_parent = mu + g*(pooled - mu)
    sigma_eps: float = 1e-6,
    n_valid: Optional[int] = None,      # children beyond this are shape-pad, not data
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, n_child, h_kv, k_dim = child_k.shape
    v_dim = child_v.shape[-1]
    c = compression_rate
    n_parent = (n_child + c - 1) // c
    pad = n_parent * c - n_child
    dtype = child_k.dtype
    if n_valid is None:
        n_valid = n_child
    assert 0 < n_valid <= n_child

    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    kf = child_k.to(compute_dtype)
    vf = child_v.to(compute_dtype)
    if pad:
        kf = torch.nn.functional.pad(kf, (0, 0, 0, 0, 0, pad))
        vf = torch.nn.functional.pad(vf, (0, 0, 0, 0, 0, pad))
    kf = kf.reshape(B, n_parent, c, h_kv, k_dim)
    vf = vf.reshape(B, n_parent, c, h_kv, v_dim)

    valid = torch.arange(n_parent * c, device=child_k.device).reshape(n_parent, c) < n_valid
    count = valid.sum(dim=1).clamp(min=1)  # [n_parent]

    if probe_w is None or probe_tau is None:
        # as recsa_build_kernel: fp32 sum over valid children / count
        k_parent = kf.sum(dim=2) / count[None, :, None, None]
        alpha = None
    elif standardize and _fused_pool_ok(child_k, child_v, c, probe_w, probe_tau, probe_gate, n_valid):
        # fused Triton path (two segments around the gemv)
        from recsa import pool_fused
        return pool_fused.pool_std_fused(child_k, child_v, probe_w, probe_tau, probe_gate, sigma_eps)
    elif standardize:
        mu = kf.sum(dim=2, keepdim=True) / count[None, :, None, None, None]  # [B,P,1,H,K]
        dev = (kf - mu).masked_fill(~valid[None, :, :, None, None], 0.0)
        denom = (count[None, :, None] * k_dim).to(compute_dtype)             # [1,P,1]
        var = dev.pow(2).sum(dim=(2, 4)) / denom                             # [B,P,H]
        sigma = var.clamp_min(sigma_eps * sigma_eps).sqrt()
        logits = probe_tau.to(compute_dtype) * (dev @ probe_w.to(compute_dtype))
        logits = logits / sigma[:, :, None, :]
        logits = logits.masked_fill(~valid[None, :, :, None], float("-inf"))
        alpha = torch.softmax(logits, dim=2)
        pooled = (alpha.unsqueeze(-1) * kf).sum(dim=2)
        mu2 = mu.squeeze(2)
        g = probe_gate.to(compute_dtype) if probe_gate is not None else 1.0
        k_parent = mu2 + g * (pooled - mu2)
    else:
        logits = probe_tau.to(compute_dtype) * (kf @ probe_w.to(compute_dtype))  # [B, P, c, H_kv]
        logits = logits.masked_fill(~valid[None, :, :, None], float("-inf"))
        alpha = torch.softmax(logits, dim=2)  # uniform over valid children at tau=0
        k_parent = (alpha.unsqueeze(-1) * kf).sum(dim=2)
    v_parent = vf.sum(dim=2) / count[None, :, None, None]  # values stay mean-pooled
    return k_parent.to(dtype), v_parent.to(dtype)


def _pool_one_level_mlp(
    child_k: torch.Tensor,  # [B, N_child, H_kv, K]
    child_v: torch.Tensor,  # [B, N_child, H_kv, V]
    compression_rate: int,
    probe_mode: str,                    # "mlp_scorer" | "mlp_pooler"
    probe_w: torch.Tensor,              # [K] fp32 (shared_std backbone)
    probe_tau: torch.Tensor,            # [] fp32
    probe_gate: Optional[torch.Tensor],  # [] fp32 (None => 1.0)
    probe_extra: Dict[str, torch.Tensor],  # mlp_w1 / mlp_b1 / mlp_w2, one layer's slice
    *,
    sigma_eps: float = 1e-6,
    n_valid: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Residual-MLP supersets of the shared_std bucket pool."""
    if probe_mode not in MLP_MODES:
        raise ValueError(f"not an MLP probe mode: {probe_mode!r}")
    if probe_mode == "mlp_pooler" and _fused_pool_ok(child_k, child_v, compression_rate, probe_w, probe_tau,
                                                     probe_gate, child_k.shape[1] if n_valid is None else n_valid):
        # fused Triton path around the three cuBLAS products
        from recsa import pool_fused
        return pool_fused.pool_mlp_fused(child_k, child_v, probe_w, probe_tau, probe_gate, probe_extra, sigma_eps)
    B, n_child, h_kv, k_dim = child_k.shape
    v_dim = child_v.shape[-1]
    c = compression_rate
    n_parent = (n_child + c - 1) // c
    pad = n_parent * c - n_child
    dtype = child_k.dtype
    if n_valid is None:
        n_valid = n_child
    assert 0 < n_valid <= n_child

    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    kf = child_k.to(compute_dtype)
    vf = child_v.to(compute_dtype)
    if pad:
        kf = torch.nn.functional.pad(kf, (0, 0, 0, 0, 0, pad))
        vf = torch.nn.functional.pad(vf, (0, 0, 0, 0, 0, pad))
    kf = kf.reshape(B, n_parent, c, h_kv, k_dim)
    vf = vf.reshape(B, n_parent, c, h_kv, v_dim)

    valid = torch.arange(n_parent * c, device=child_k.device).reshape(n_parent, c) < n_valid
    count = valid.sum(dim=1).clamp(min=1)  # [n_parent]

    mu = kf.sum(dim=2, keepdim=True) / count[None, :, None, None, None]      # [B,P,1,H,K]
    dev = (kf - mu).masked_fill(~valid[None, :, :, None, None], 0.0)
    denom = (count[None, :, None] * k_dim).to(compute_dtype)                 # [1,P,1]
    var = dev.pow(2).sum(dim=(2, 4)) / denom                                 # [B,P,H]
    sigma = var.clamp_min(sigma_eps * sigma_eps).sqrt()
    tau = probe_tau.to(compute_dtype)
    logits = tau * (dev @ probe_w.to(compute_dtype))
    logits = logits / sigma[:, :, None, :]                                   # [B,P,c,H]
    w1 = probe_extra["mlp_w1"].to(compute_dtype)
    b1 = probe_extra["mlp_b1"].to(compute_dtype)
    w2 = probe_extra["mlp_w2"].to(compute_dtype)
    if probe_mode == "mlp_scorer":
        z = dev / sigma[:, :, None, :, None]                                 # [B,P,c,H,K]
        hid = torch.nn.functional.silu(z @ w1.transpose(0, 1) + b1)          # [B,P,c,H,H1]
        logits = logits + tau * (hid @ w2)                                   # [B,P,c,H]
    logits = logits.masked_fill(~valid[None, :, :, None], float("-inf"))
    alpha = torch.softmax(logits, dim=2)
    pooled = (alpha.unsqueeze(-1) * kf).sum(dim=2)                           # [B,P,H,K]
    mu2 = mu.squeeze(2)
    g = probe_gate.to(compute_dtype) if probe_gate is not None else 1.0
    k_parent = mu2 + g * (pooled - mu2)
    if probe_mode == "mlp_pooler":
        delta = (pooled - mu2) / sigma[..., None]                            # [B,P,H,K]
        rho = mu2.pow(2).mean(dim=-1).clamp_min(sigma_eps * sigma_eps).sqrt()  # [B,P,H]
        h = torch.cat([mu2 / rho[..., None], delta], dim=-1)                 # [B,P,H,2K]
        hid = torch.nn.functional.silu(h @ w1.transpose(0, 1) + b1)          # [B,P,H,H1]
        k_parent = k_parent + sigma[..., None] * (hid @ w2.transpose(0, 1))
    v_parent = vf.sum(dim=2) / count[None, :, None, None]  # values stay mean-pooled
    return k_parent.to(dtype), v_parent.to(dtype)


class DecodeBucketPool:
    """Decode-shape specialisation of `_pool_one_level(standardize=True)`: B=1, one parent bucket, c children."""

    __slots__ = ("c", "d", "eps2", "kb", "vb", "invalid", "cnt5", "cnt4", "den3")

    def __init__(self, h_kv: int, d: int, c: int, device, sigma_eps: float = 1e-6):
        self.c, self.d, self.eps2 = c, d, sigma_eps * sigma_eps
        f32 = dict(device=device, dtype=torch.float32)
        self.kb = torch.zeros(1, 1, c, h_kv, d, **f32)
        self.vb = torch.zeros(1, 1, c, h_kv, d, **f32)
        ar = torch.arange(c, device=device).reshape(1, c)
        self.invalid = [None] + [~(ar < n) for n in range(1, c + 1)]
        self.cnt5 = [None] + [torch.full((1, 1, 1, 1, 1), float(n), **f32)
                              for n in range(1, c + 1)]
        self.cnt4 = [None] + [torch.full((1, 1, 1, 1), float(n), **f32)
                              for n in range(1, c + 1)]
        self.den3 = [None] + [torch.full((1, 1, 1), float(n * d), **f32)
                              for n in range(1, c + 1)]

    def __call__(self, child_k, child_v, probe_w, probe_tau, probe_gate):
        """Returns fp32 [1, 1, H_kv, D] k/v parents; the caller casts them into the tree slot."""
        n = child_k.shape[1]
        full = n == self.c
        kb, vb = self.kb, self.vb
        kb[:, 0, :n].copy_(child_k)   # bf16 -> fp32 widening is exact
        vb[:, 0, :n].copy_(child_v)
        if not full:                  # stale tail from a previous, wider call
            kb[:, 0, n:].zero_()
            vb[:, 0, n:].zero_()
        mu = kb.sum(dim=2, keepdim=True) / self.cnt5[n]
        dev = kb - mu
        if not full:
            dev = dev.masked_fill(self.invalid[n][None, :, :, None, None], 0.0)
        var = dev.pow(2).sum(dim=(2, 4)) / self.den3[n]
        sigma = var.clamp_min(self.eps2).sqrt()
        logits = probe_tau * (dev @ probe_w)
        logits = logits / sigma[:, :, None, :]
        if not full:
            logits = logits.masked_fill(self.invalid[n][None, :, :, None],
                                        float("-inf"))
        alpha = torch.softmax(logits, dim=2)
        pooled = (alpha.unsqueeze(-1) * kb).sum(dim=2)
        mu2 = mu.squeeze(2)
        g = 1.0 if probe_gate is None else probe_gate
        return mu2 + g * (pooled - mu2), vb.sum(dim=2) / self.cnt4[n]


class DecodeBucketPoolMLP:
    """Decode-shape specialisation of `_pool_one_level_mlp` (B=1, one parent bucket, c children) for the MLP probe
    modes."""

    __slots__ = ("mode", "c", "d", "eps2", "kb", "vb", "invalid",
                 "cnt5", "cnt4", "den3", "zero5", "_scr", "_xscr")

    def __init__(self, probe_mode: str, h_kv: int, d: int, c: int, device,
                 sigma_eps: float = 1e-6):
        if probe_mode not in MLP_MODES:
            raise ValueError(f"not an MLP probe mode: {probe_mode!r}")
        self.mode, self.c, self.d = probe_mode, c, d
        self.eps2 = sigma_eps * sigma_eps
        f32 = dict(device=device, dtype=torch.float32)
        self.kb = torch.zeros(1, 1, c, h_kv, d, **f32)
        self.vb = torch.zeros(1, 1, c, h_kv, d, **f32)
        self.zero5 = torch.zeros(1, 1, 1, 1, 1, **f32)
        ar = torch.arange(c, device=device).reshape(1, c)
        self.invalid = [None] + [~(ar < n) for n in range(1, c + 1)]
        self.cnt5 = [None] + [torch.full((1, 1, 1, 1, 1), float(n), **f32)
                              for n in range(1, c + 1)]
        self.cnt4 = [None] + [torch.full((1, 1, 1, 1), float(n), **f32)
                              for n in range(1, c + 1)]
        self.den3 = [None] + [torch.full((1, 1, 1), float(n * d), **f32)
                              for n in range(1, c + 1)]
        self._scr: Dict[torch.dtype, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._xscr: dict = {}

    def __call__(self, child_k, child_v, probe_w, probe_tau, probe_gate,
                 probe_extra):
        """Returns fp32 [1, 1, H_kv, D] k/v parents; the caller casts them into the tree slot."""
        n = child_k.shape[1]
        full = n == self.c
        kb, vb = self.kb, self.vb
        kb[:, 0, :n].copy_(child_k)   # bf16 -> fp32 widening is exact
        vb[:, 0, :n].copy_(child_v)
        if not full:                  # stale tail from a previous, wider call
            kb[:, 0, n:].zero_()
            vb[:, 0, n:].zero_()
        mu = kb.sum(dim=2, keepdim=True) / self.cnt5[n]
        dev = kb - mu
        if not full:
            dev = dev.masked_fill(self.invalid[n][None, :, :, None, None], 0.0)
        var = dev.pow(2).sum(dim=(2, 4)) / self.den3[n]
        sigma = var.clamp_min(self.eps2).sqrt()
        logits = probe_tau * (dev @ probe_w)
        logits = logits / sigma[:, :, None, :]
        w1, b1, w2 = (probe_extra["mlp_w1"], probe_extra["mlp_b1"],
                      probe_extra["mlp_w2"])
        if self.mode == "mlp_scorer":
            z = dev / sigma[:, :, None, :, None]
            hid = torch.nn.functional.silu(z @ w1.transpose(0, 1) + b1)
            logits = logits + probe_tau * (hid @ w2)
        if not full:
            logits = logits.masked_fill(self.invalid[n][None, :, :, None],
                                        float("-inf"))
        alpha = torch.softmax(logits, dim=2)
        pooled = (alpha.unsqueeze(-1) * kb).sum(dim=2)
        mu2 = mu.squeeze(2)
        g = 1.0 if probe_gate is None else probe_gate
        k_parent = mu2 + g * (pooled - mu2)
        if self.mode == "mlp_pooler":
            delta = (pooled - mu2) / sigma[..., None]
            rho = mu2.pow(2).mean(dim=-1).clamp_min(self.eps2).sqrt()
            h = torch.cat([mu2 / rho[..., None], delta], dim=-1)
            hid = torch.nn.functional.silu(h @ w1.transpose(0, 1) + b1)
            k_parent = k_parent + sigma[..., None] * (hid @ w2.transpose(0, 1))
        return k_parent, vb.sum(dim=2) / self.cnt4[n]

    def cursor(self, child_k, child_v, win, probe_w, probe_tau, probe_gate,
               probe_extra, out_k, out_v) -> None:
        """CUDA-graph version of __call__ plus the slot assignment, for the open bucket of a BucketCursor window:
        child_k/child_v are the level-(l-1) growth buffers, out_k/out_v the level-l ones."""
        kb = torch.where(win.invalid5, self.zero5,
                         child_k[:, win.idx].unsqueeze(1))
        vb = torch.where(win.invalid5, self.zero5,
                         child_v[:, win.idx].unsqueeze(1))
        mu = kb.sum(dim=2, keepdim=True) / win.cnt5
        dev = (kb - mu).masked_fill_(win.invalid5, 0.0)
        var = dev.pow(2).sum(dim=(2, 4)) / win.den3
        sigma = var.clamp_min(self.eps2).sqrt()
        logits = probe_tau * (dev @ probe_w)
        logits = logits / sigma[:, :, None, :]
        w1, b1, w2 = (probe_extra["mlp_w1"], probe_extra["mlp_b1"],
                      probe_extra["mlp_w2"])
        if self.mode == "mlp_scorer":
            z = dev / sigma[:, :, None, :, None]
            hid = torch.nn.functional.silu(z @ w1.transpose(0, 1) + b1)
            logits = logits + probe_tau * (hid @ w2)
        alpha = torch.softmax(logits.masked_fill_(win.invalid4, float("-inf")),
                              dim=2)
        pooled = (alpha.unsqueeze(-1) * kb).sum(dim=2)
        mu2 = mu.squeeze(2)
        diff = pooled - mu2
        g = 1.0 if probe_gate is None else probe_gate
        k_parent = mu2 + g * diff
        if self.mode == "mlp_pooler":
            delta = diff / sigma[..., None]
            rho = mu2.pow(2).mean(dim=-1).clamp_min(self.eps2).sqrt()
            h = torch.cat([mu2 / rho[..., None], delta], dim=-1)
            hid = torch.nn.functional.silu(h @ w1.transpose(0, 1) + b1)
            k_parent = k_parent + sigma[..., None] * (hid @ w2.transpose(0, 1))
        v_parent = vb.sum(dim=2) / win.cnt4
        out_k.index_copy_(1, win.p, k_parent.to(out_k.dtype))
        out_v.index_copy_(1, win.p, v_parent.to(out_v.dtype))

    def _scratch(self, dtype: torch.dtype, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """cursor_k's persistent operands: h [1,1,H,2D] fp32 (the cat target) and the open parent's K row in the tree
        dtype."""
        s = self._scr.get(dtype)
        if s is None:
            if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("DecodeBucketPoolMLP.cursor_k scratch must be "
                                   "allocated before CUDA-graph capture (warm step)")
            h_kv = self.kb.shape[3]
            s = self._scr[dtype] = (
                torch.empty(1, 1, h_kv, 2 * self.d, device=device, dtype=torch.float32),
                torch.empty(1, 1, h_kv, self.d, device=device, dtype=dtype))
        return s

    def cursor_k(self, child_k, win, probe_w, tau: float, gate: float,
                 probe_extra, out_k) -> None:
        """The K half of cursor(): writes the open parent's key only."""

        h, kpb = self._scratch(out_k.dtype, out_k.device)
        d = self.d
        kb = child_k[:, win.idx_z].unsqueeze(1).float()
        mu = kb.sum(dim=2, keepdim=True) / win.cnt5
        dev = (kb - mu).masked_fill_(win.invalid5, 0.0)
        var = dev.pow(2).sum(dim=(2, 4)) / win.den3
        sigma = var.clamp_min(self.eps2).sqrt()
        logits = tau * (dev @ probe_w)
        logits = logits / sigma[:, :, None, :]
        w1, b1, w2 = (probe_extra["mlp_w1"], probe_extra["mlp_b1"],
                      probe_extra["mlp_w2"])
        if self.mode == "mlp_scorer":
            z = dev / sigma[:, :, None, :, None]
            hid = torch.nn.functional.silu(z @ w1.transpose(0, 1) + b1)
            logits = logits + tau * (hid @ w2)
        alpha = torch.softmax(logits.masked_fill_(win.invalid4, float("-inf")),
                              dim=2)
        pooled = (alpha.unsqueeze(-1) * kb).sum(dim=2)
        mu2 = mu.squeeze(2)
        diff = pooled - mu2
        if self.mode == "mlp_pooler":
            k_parent = mu2 + gate * diff
            torch.div(diff, sigma[..., None], out=h[..., d:])
            rho = mu2.pow(2).mean(dim=-1).clamp_min(self.eps2).sqrt()
            torch.div(mu2, rho[..., None], out=h[..., :d])
            hid = torch.nn.functional.silu(h @ w1.transpose(0, 1) + b1)
            torch.add(k_parent, sigma[..., None] * (hid @ w2.transpose(0, 1)),
                      out=kpb)
        else:
            torch.add(mu2, gate * diff, out=kpb)
        out_k.index_copy_(1, win.p, kpb)

    def cursor_k_exact(self, child_k, win, probe_w, tau: float, gate: float,
                       probe_extra, out_k, leaf=None) -> None:
        """cursor_k (mlp_pooler) as exact-order Triton segments around the three cuBLAS products."""
        from recsa.mlp_pool_exact import ExactScratch, pool_k_exact
        h1 = probe_extra["mlp_w1"].shape[0]
        key = (out_k.device, h1)
        scr = self._xscr.get(key)
        if scr is None:
            scr = self._xscr[key] = ExactScratch(self.kb.shape[3], self.d, self.c, h1,
                                                 out_k.device)
        pool_k_exact(scr, child_k, win, probe_w, tau, gate, probe_extra, out_k, leaf)

    def cursor_k_fused(self, child_k, win, probe_w, tau: float, gate: float,
                       probe_extra, out_k, leaf=None, win2=None, out2_k=None,
                       wt=None) -> None:
        """cursor_k_exact as one Triton launch, with the three cuBLAS products computed in-kernel in cuBLAS's
        accumulation orders."""

        from recsa.mlp_pool_exact import ExactScratch, pool_k_fused
        h1 = probe_extra["mlp_w1"].shape[0]
        key = (out_k.device, h1)
        scr = self._xscr.get(key)
        if scr is None:
            scr = self._xscr[key] = ExactScratch(self.kb.shape[3], self.d, self.c, h1,
                                                 out_k.device)
        pool_k_fused(scr, child_k, win, probe_w, tau, gate, probe_extra, out_k, leaf,
                     win2, out2_k, wt)


class _BucketWindow(NamedTuple):
    idx: torch.Tensor
    p: torch.Tensor         # [1] int64: the open parent's slot
    invalid5: torch.Tensor  # [1,1,c,1,1] bool: j >= n
    invalid4: torch.Tensor  # [1,1,c,1]   bool
    cnt5: torch.Tensor      # [1,1,1,1,1] fp32: float(n)
    cnt4: torch.Tensor      # [1,1,1,1]   fp32
    den3: torch.Tensor      # [1,1,1]     fp32: float(n * d)
    idx_z: Optional[torch.Tensor] = None


_cursor_kernel = None


def _bucket_cursor_kernel():
    global _cursor_kernel
    if _cursor_kernel is None:  # lazy: keeps CPU-only imports off triton
        from recsa.parallel import recsa_bucket_cursor_kernel
        _cursor_kernel = recsa_bucket_cursor_kernel
    return _cursor_kernel


class BucketCursor:
    """Open-bucket windows of every pooled level as device tensors, derived from a decode session's cursor, so that
    DecodeBucketPoolMLP.cursor runs inside a CUDA graph."""

    __slots__ = ("t_dev", "c", "d", "capacity", "offs", "_wins", "fused",
                 "_fbuf", "_fwins")

    def __init__(self, t_dev: torch.Tensor, c: int, d: int, capacity: int,
                 fused: Optional[bool] = None):
        if c * d >= 1 << 24:
            raise ValueError("den3 = n*d must be exact in fp32")
        self.t_dev, self.c, self.d, self.capacity = t_dev, c, d, capacity
        self.offs = torch.arange(c, device=t_dev.device)
        self._wins: List[Optional[_BucketWindow]] = []
        self.fused = (RECSA_MLP_CURSOR_POOL == "v2") if fused is None else bool(fused)
        self._fbuf = None
        self._fwins: List[Optional[_BucketWindow]] = []

    def spare_row(self, level: int) -> int:
        """Spare (never-written) row of the level-(level-1) child buffer."""
        cp = self.c ** (level - 1)
        return (self.capacity + cp - 1) // cp

    def refresh(self, n_levels: int) -> None:
        """Recompute the windows of levels 1..n_levels-1 from the cursor."""
        if self.fused and self.t_dev.is_cuda:
            self._refresh_fused(n_levels)
            return
        c, t = self.c, self.t_dev
        wins: List[Optional[_BucketWindow]] = [None]
        for level in range(1, n_levels):
            cp = c ** (level - 1)
            q = t if cp == 1 else torch.div(t, cp, rounding_mode="floor")
            p = torch.div(q, c, rounding_mode="floor")
            raw = torch.add(self.offs, p, alpha=c)            # lo + j
            invalid = raw > q
            cnt = (raw <= q).sum(dtype=torch.float32)         # float(n), exact
            wins.append(_BucketWindow(
                raw.clamp_max((self.capacity - 1) // cp), p,
                invalid.view(1, 1, c, 1, 1), invalid.view(1, 1, c, 1),
                cnt.view(1, 1, 1, 1, 1), cnt.view(1, 1, 1, 1),
                (cnt * self.d).view(1, 1, 1),
                torch.where(invalid, self.spare_row(level), raw)
                if self.fused else None))
        self._wins = wins

    def _refresh_fused(self, n_levels: int) -> None:
        n_pooled = n_levels - 1
        if n_pooled < 1:
            self._wins = [None]
            return
        if self._fbuf is None or len(self._fwins) - 1 < n_pooled:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("BucketCursor: fused window buffers must be "
                                   "allocated before CUDA-graph capture")
            c, dev = self.c, self.t_dev.device
            i64 = torch.zeros(n_pooled, 2 * c + 1, dtype=torch.int64, device=dev)
            u8 = torch.zeros(n_pooled, c, dtype=torch.uint8, device=dev)
            f32 = torch.zeros(n_pooled, 2, dtype=torch.float32, device=dev)
            inv = u8.view(torch.bool)
            wins: List[Optional[_BucketWindow]] = [None]
            for lv in range(n_pooled):
                wins.append(_BucketWindow(
                    i64[lv, :c], i64[lv, 2 * c:], inv[lv].view(1, 1, c, 1, 1),
                    inv[lv].view(1, 1, c, 1), f32[lv, :1].view(1, 1, 1, 1, 1),
                    f32[lv, :1].view(1, 1, 1, 1), f32[lv, 1:].view(1, 1, 1),
                    i64[lv, c:2 * c]))
            self._fbuf, self._fwins = (i64, u8, f32), wins
        i64, u8, f32 = self._fbuf
        with torch.cuda.device(self.t_dev.device):
            _bucket_cursor_kernel()[(1,)](
                self.t_dev, i64, u8, f32, self.capacity,
                C=self.c, D=self.d, N_POOLED=n_pooled)
        self._wins = self._fwins[:n_levels]

    def __getitem__(self, level: int) -> _BucketWindow:
        return self._wins[level]


_BUCKET_POOLS: dict = {}


def decode_bucket_pool(h_kv: int, d: int, c: int, device) -> DecodeBucketPool:
    key = (device.type, device.index, h_kv, d, c)
    p = _BUCKET_POOLS.get(key)
    if p is None:
        p = _BUCKET_POOLS[key] = DecodeBucketPool(h_kv, d, c, device)
    return p


def decode_bucket_pool_mlp(probe_mode: str, h_kv: int, d: int, c: int,
                           device) -> DecodeBucketPoolMLP:
    key = (probe_mode, device.type, device.index, h_kv, d, c)
    p = _BUCKET_POOLS.get(key)
    if p is None:
        p = _BUCKET_POOLS[key] = DecodeBucketPoolMLP(probe_mode, h_kv, d, c,
                                                     device)
    return p


def build_tree_layers(
    k: torch.Tensor,  # [B, T, H_kv, K]
    v: torch.Tensor,  # [B, T, H_kv, V]
    compression_rate: int,
    max_top_nodes: int,
    probe_w: Optional[torch.Tensor] = None,  # per_level: [L_max, K]; shared_std: [K]
    probe_tau: Optional[torch.Tensor] = None,  # per_level: [L_max]; shared_std: []
    probe_mode: str = "per_level",
    probe_gate: Optional[torch.Tensor] = None,  # shared_std only: [] (None => 1.0)
    build_chunk_parents: Optional[int] = None,  # execution tile: parent buckets per slice
    probe_extra: Optional[Dict[str, torch.Tensor]] = None,  # MLP modes: one layer's slice
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Build the RecSA K/V pyramid in torch. probe_w/probe_tau None => mean pooling."""

    if build_chunk_parents is not None and build_chunk_parents < 1:
        raise ValueError(f"build_chunk_parents must be >= 1, got {build_chunk_parents}")
    num_layers = compute_num_layers(k.shape[1], compression_rate, max_top_nodes)
    if probe_mode not in {"per_level", "shared_std"} and probe_mode not in MLP_MODES:
        raise ValueError(f"unknown probe_mode {probe_mode!r}")
    mlp = probe_mode in MLP_MODES
    shared = probe_mode == "shared_std" or mlp  # mlp modes share params across levels
    if mlp:
        if probe_w is None:
            raise ValueError(f"{probe_mode} requires probe params")
        if build_chunk_parents is not None:
            raise ValueError(f"{probe_mode} does not support build_chunk_parents "
                             "(lengths below 5M run untiled; tiling is unchecked here)")
        validate_probe_extra(probe_mode, probe_extra, k.shape[-1])
    elif probe_extra:
        raise ValueError(f"probe_extra given but mode is {probe_mode!r}")
    if probe_w is not None:
        if probe_tau is None:
            raise ValueError("probe_w requires probe_tau")
        if shared:
            if probe_w.dim() != 1:
                raise ValueError(f"shared_std expects probe_w [K], got {tuple(probe_w.shape)}")
            if probe_tau.dim() != 0:
                raise ValueError(f"shared_std expects scalar probe_tau, got {tuple(probe_tau.shape)}")
            if probe_gate is not None and probe_gate.dim() != 0:
                raise ValueError(f"shared_std expects scalar probe_gate, got {tuple(probe_gate.shape)}")
        elif probe_w.shape[0] < num_layers - 1 or probe_tau.shape[0] < num_layers - 1:
            raise ValueError(
                f"probe params cover {probe_w.shape[0]} levels; sequence needs {num_layers - 1}"
            )
    layers_k, layers_v = [k], [v]
    for level in range(1, num_layers):
        if probe_w is None:
            w_l = tau_l = None
        elif shared:
            w_l, tau_l = probe_w, probe_tau
        else:
            w_l, tau_l = probe_w[level - 1], probe_tau[level - 1]
        child_k, child_v = layers_k[-1], layers_v[-1]
        step = None if build_chunk_parents is None else build_chunk_parents * compression_rate
        if mlp:
            nk, nv = _pool_one_level_mlp(child_k, child_v, compression_rate,
                                         probe_mode, w_l, tau_l, probe_gate, probe_extra)
        elif step is None or child_k.shape[1] <= step:
            nk, nv = _pool_one_level(child_k, child_v, compression_rate, w_l, tau_l,
                                     standardize=shared, probe_gate=probe_gate if shared else None)
        else:
            n = child_k.shape[1]
            parts_k, parts_v = [], []
            for a in range(0, n, step):
                b = min(a + step, n)
                ck, cv = child_k[:, a:b], child_v[:, a:b]
                if b - a < step:
                    ck = torch.nn.functional.pad(ck, (0, 0, 0, 0, 0, step - (b - a)))
                    cv = torch.nn.functional.pad(cv, (0, 0, 0, 0, 0, step - (b - a)))
                pk, pv = _pool_one_level(ck, cv, compression_rate, w_l, tau_l,
                                         standardize=shared,
                                         probe_gate=probe_gate if shared else None,
                                         n_valid=(b - a) if b - a < step else None)
                keep = (b - a + compression_rate - 1) // compression_rate
                parts_k.append(pk[:, :keep])
                parts_v.append(pv[:, :keep])
            nk = torch.cat(parts_k, dim=1)
            nv = torch.cat(parts_v, dim=1)
        layers_k.append(nk)
        layers_v.append(nv)
    return layers_k, layers_v
