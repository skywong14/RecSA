"""Decode-time incremental RecSA maintenance."""
from __future__ import annotations

import os
import weakref
from typing import List, Optional, Tuple

import torch

from recsa.probe_pool import (
    MLP_MODES,
    _pool_one_level,
    _pool_one_level_mlp,
    build_tree_layers,
    compute_num_layers,
    decode_bucket_pool,
    decode_bucket_pool_mlp,
)

FUSED_POOL_DEFAULT = os.environ.get("RECSA_FUSED_POOL", "1").lower() in {"1", "true", "yes"}

MLP_OFFLOAD_GRAPH = os.environ.get("RECSA_MLP_OFFLOAD_GRAPH", "on").strip().lower()
if MLP_OFFLOAD_GRAPH not in ("on", "off"):
    raise ValueError("RECSA_MLP_OFFLOAD_GRAPH must be 'on' or 'off', "
                     f"got {MLP_OFFLOAD_GRAPH!r}")

OFFLOAD_POOL_V2 = os.environ.get("RECSA_OFFLOAD_POOL_V2", "off").strip().lower()
if OFFLOAD_POOL_V2 not in ("on", "off"):
    raise ValueError("RECSA_OFFLOAD_POOL_V2 must be 'on' or 'off', "
                     f"got {OFFLOAD_POOL_V2!r}")

_pyramid_build = None
_build_kernel = None
_probe_pool_launch = None
_leaf_store_kernel = None
_leaf_store_bucket_kernel = None
_mirror_store_kernel = None


def _kernel_build():
    global _pyramid_build
    if _pyramid_build is None:
        from recsa.parallel import recsa_build_pyramid
        _pyramid_build = recsa_build_pyramid
    return _pyramid_build


def _leaf_store():
    global _leaf_store_kernel
    if _leaf_store_kernel is None:
        from recsa.parallel import recsa_leaf_store_kernel
        _leaf_store_kernel = recsa_leaf_store_kernel
    return _leaf_store_kernel


def _leaf_store_bucket():
    global _leaf_store_bucket_kernel
    if _leaf_store_bucket_kernel is None:
        from recsa.parallel import recsa_leaf_store_bucket_kernel
        _leaf_store_bucket_kernel = recsa_leaf_store_bucket_kernel
    return _leaf_store_bucket_kernel


def _mirror_store():
    global _mirror_store_kernel
    if _mirror_store_kernel is None:
        from recsa.parallel import recsa_leaf_mirror_store_kernel
        _mirror_store_kernel = recsa_leaf_mirror_store_kernel
    return _mirror_store_kernel


def _host_unregister_quiet(ptr: int) -> None:
    """Finalizer for registered pool tensors."""
    try:
        torch.cuda.cudart().cudaHostUnregister(ptr)
    except Exception:
        pass


def pinned_host_zeros(shape, dtype) -> torch.Tensor:
    """A zeroed CPU tensor page-locked in place via cudaHostRegister, instead of torch's pinned caching allocator."""
    t = torch.zeros(*shape, dtype=dtype)
    torch.cuda.init()
    torch.cuda.check_error(
        torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.nbytes, 0))
    weakref.finalize(t, _host_unregister_quiet, t.data_ptr())
    return t


def _raw_build_kernel():
    global _build_kernel
    if _build_kernel is None:
        from recsa.parallel import recsa_build_kernel
        _build_kernel = recsa_build_kernel
    return _build_kernel


def _fused_pool_launch():
    global _probe_pool_launch
    if _probe_pool_launch is None:
        from recsa.parallel import probe_pool_bucket
        _probe_pool_launch = probe_pool_bucket
    return _probe_pool_launch


class IncrementalTreeState:
    """Append-only mirror of `build_tree_layers` for decode."""

    def __init__(
        self,
        k: torch.Tensor,  # [B, T0, H_kv, K] prefill keys
        v: torch.Tensor,  # [B, T0, H_kv, V] prefill values
        compression_rate: int,
        max_top_nodes: int,
        *,
        probe_w: Optional[torch.Tensor] = None,
        probe_tau: Optional[torch.Tensor] = None,
        probe_mode: str = "per_level",
        probe_gate: Optional[torch.Tensor] = None,
        probe_extra=None,  # MLP modes: one layer's param dict
        reserve: int = 256,
        build_chunk_parents: Optional[int] = None,  # init build only; appends are single-bucket
        gate_b: Optional[float] = None,  # level bias: this layer's b gate (pre-clamp)
    ) -> None:
        if k.dim() != 4 or v.dim() != 4 or k.shape[1] < 1:
            raise ValueError(f"expected [B, T>=1, H, D] k/v, got {tuple(k.shape)}")
        if probe_w is None and k.is_cuda:
            with torch.cuda.device(k.device):
                layers_k, layers_v = _kernel_build()(
                    k.contiguous(), v.contiguous(), compression_rate, max_top_nodes)
        else:
            layers_k, layers_v = build_tree_layers(
                k, v, compression_rate, max_top_nodes,
                probe_w=probe_w, probe_tau=probe_tau, probe_mode=probe_mode,
                probe_gate=probe_gate, build_chunk_parents=build_chunk_parents,
                probe_extra=probe_extra,
            )
        self._setup(layers_k, layers_v, compression_rate, max_top_nodes,
                    probe_w, probe_tau, probe_mode, probe_gate, reserve,
                    probe_extra=probe_extra, gate_b=gate_b)

    @classmethod
    def from_layers(
        cls,
        layers_k: List[torch.Tensor],
        layers_v: List[torch.Tensor],
        compression_rate: int,
        max_top_nodes: int,
        *,
        probe_w: Optional[torch.Tensor] = None,
        probe_tau: Optional[torch.Tensor] = None,
        probe_mode: str = "per_level",
        probe_gate: Optional[torch.Tensor] = None,
        probe_extra=None,  # MLP modes: one layer's param dict
        reserve: int = 256,
        buffers=None,
        gate_b: Optional[float] = None,  # level bias: this layer's b gate (pre-clamp)
    ) -> "IncrementalTreeState":
        """Adopt an already-built tree (e.g. a streamed-prefill pyramid)."""
        if not layers_k or len(layers_k) != len(layers_v):
            raise ValueError("layers_k/layers_v must be non-empty, same length")
        t0 = layers_k[0].shape[1]
        if t0 < 1:
            raise ValueError("empty leaf level")
        want = compute_num_layers(t0, compression_rate, max_top_nodes)
        if len(layers_k) != want:
            raise ValueError(
                f"{len(layers_k)} levels but T={t0} needs {want} at "
                f"(c={compression_rate}, mtn={max_top_nodes})"
            )
        n = t0
        for lvl, (lk, lv) in enumerate(zip(layers_k, layers_v)):
            if lvl:
                n = (n + compression_rate - 1) // compression_rate
            if lk.dim() != 4 or lv.dim() != 4 or lk.shape[1] != n or lv.shape[1] != n:
                raise ValueError(f"level {lvl}: width != ceil-chain {n} or not 4-dim")
            for t, r in ((lk, layers_k[0]), (lv, layers_v[0])):
                if (t.shape[0] != r.shape[0] or t.shape[2] != r.shape[2]
                        or t.shape[3] != r.shape[3] or t.dtype != r.dtype
                        or t.device != r.device):
                    raise ValueError(
                        f"level {lvl}: (B,H,D,dtype,device) not uniform with leaves")
        self = cls.__new__(cls)
        self._setup(layers_k, layers_v, compression_rate, max_top_nodes,
                    probe_w, probe_tau, probe_mode, probe_gate, reserve,
                    buffers=buffers, probe_extra=probe_extra, gate_b=gate_b)
        return self

    @classmethod
    def preallocate(
        cls,
        n_states: int,
        t0: int,
        compression_rate: int,
        max_top_nodes: int,
        *,
        b: int,
        h_kv: int,
        k_dim: int,
        v_dim: int,
        dtype,
        device,
        reserve: int = 256,
        offload_leaves: bool = False,
    ) -> List[List[torch.Tensor]]:
        """Allocate every growth buffer `n_states` adopters will need, up front."""
        n_lvl = compute_num_layers(t0 + reserve, compression_rate, max_top_nodes)
        out = []
        for _ in range(n_states):
            ks, vs = [], []
            for lvl in range(n_lvl):
                cap = ((t0 + reserve + compression_rate ** lvl - 1)
                       // (compression_rate ** lvl)) + 1
                if lvl == 0 and offload_leaves:
                    # registered, not pin_memory=True (see pinned_host_zeros)
                    ks.append(pinned_host_zeros((b, cap, h_kv, k_dim), dtype))
                    vs.append(pinned_host_zeros((b, cap, h_kv, v_dim), dtype))
                else:
                    ks.append(torch.zeros(b, cap, h_kv, k_dim, dtype=dtype, device=device))
                    vs.append(torch.zeros(b, cap, h_kv, v_dim, dtype=dtype, device=device))
            out.append([ks, vs])
        return out

    def _setup(self, layers_k, layers_v, compression_rate, max_top_nodes,
               probe_w, probe_tau, probe_mode, probe_gate, reserve,
               buffers=None, probe_extra=None, gate_b=None) -> None:
        if reserve < 0:
            raise ValueError(f"reserve must be >= 0, got {reserve}")
        self.c = compression_rate
        self.mtn = max_top_nodes
        self._probe_w = probe_w
        self._probe_tau = probe_tau
        self._probe_gate = probe_gate
        self._probe_extra = probe_extra
        if probe_mode not in {"per_level", "shared_std"} and probe_mode not in MLP_MODES:
            raise ValueError(f"unknown probe_mode {probe_mode!r} "
                             f"(valid: per_level, shared_std, {sorted(MLP_MODES)})")
        self._mode = probe_mode
        self._shared = probe_mode == "shared_std"
        if probe_mode in MLP_MODES and not probe_extra:
            raise ValueError(f"{probe_mode} state requires probe_extra")
        self._fused_ok: Optional[bool] = None  # shape/dtype eligibility, cached on first refresh
        self._mirror = None
        self._capacity = layers_k[0].shape[1] + reserve

        max_layers = compute_num_layers(self._capacity, self.c, self.mtn)
        if probe_w is not None and probe_mode == "per_level" and (
            probe_w.shape[0] < max_layers - 1 or probe_tau.shape[0] < max_layers - 1
        ):
            raise ValueError(
                f"per_level probe covers {probe_w.shape[0]} levels; capacity "
                f"{self._capacity} tokens can need {max_layers - 1}"
            )
        self._k: List[torch.Tensor] = []
        self._v: List[torch.Tensor] = []
        self._sizes: List[int] = []
        for lvl, (lk, lv) in enumerate(zip(layers_k, layers_v)):
            self._alloc_level(lvl, lk, lv,
                              buf=None if buffers is None
                              else (buffers[0][lvl], buffers[1][lvl]))
        self._offload = (not self._k[0].is_cuda and len(self._k) > 1
                         and self._k[1].is_cuda)
        self._dev = self._k[-1].device  # kernel-launch device (top level)
        if self._offload:
            if not self._k[0].is_pinned():
                raise ValueError(
                    "offloaded leaves must be pinned host memory (the kernels "
                    "read/write the pool via UVA; pageable memory is illegal)")
            if self._k[0].shape[0] != 1:
                raise ValueError("leaf offload requires B == 1")
            if len(self._sizes) < 2:
                raise ValueError(
                    "leaf offload requires >= 2 levels at adoption: a later "
                    "level-1 birth would pool HOST children through the torch "
                    "path — wrong builder AND wrong device")
            if (self._probe_w is not None and self._mode not in MLP_MODES
                    and not (self._shared and FUSED_POOL_DEFAULT
                             and self._fused_eligible())):
                raise ValueError(
                    "probe-pooled leaf offload requires the fused pool "
                    "(shared_std + RECSA_FUSED_POOL=1 + eligible shapes): the "
                    "torch bucket pool would compute on CPU leaves — wrong "
                    "builder numerics, not just slow")
        self._init_bias(layers_k, gate_b)

    # -- public ------------------------------------------------------------

    @property
    def seq_len(self) -> int:
        return self._sizes[0]

    @property
    def num_layers(self) -> int:
        return len(self._sizes)

    def layers(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Current tree as views, `build_tree_layers` ordering (level 0 = leaves)."""
        ks = [buf[:, :n] for buf, n in zip(self._k, self._sizes)]
        vs = [buf[:, :n] for buf, n in zip(self._v, self._sizes)]
        return ks, vs

    # -- incremental per-node selection bias b (level bias) -----------------

    def _bias_cap(self, level: int) -> int:
        return (self._capacity + self.c ** level - 1) // (self.c ** level) + 1

    @torch.no_grad()
    def _init_bias(self, layers_k, gate_b) -> None:
        self._g_b = 0.0
        self._b: List[Optional[torch.Tensor]] = []
        if gate_b is None:
            return
        g = min(max(float(gate_b), 0.0), 1.0)
        self._g_b = g
        if g == 0.0:
            return
        if not (self._shared or self._mode == "mlp_pooler"):
            raise ValueError(
                f"gate_b requires a shared_std α backbone "
                f"(shared_std or mlp_pooler), got {self._mode!r}")
        if layers_k[0].shape[0] != 1:
            raise ValueError("gate_b maintenance requires B == 1")
        from recsa.level_bias import compose_level_bias
        lb = compose_level_bias(
            [lk for lk in layers_k], layers_k[0].shape[1],
            self._probe_w, self._probe_tau, g, self.c)
        assert lb is not None and len(lb) == len(layers_k)
        H_kv = layers_k[0].shape[2]
        self._b = [None]
        for lvl in range(1, len(layers_k)):
            buf = torch.zeros(1, self._bias_cap(lvl), H_kv,
                              dtype=torch.float32,
                              device=self._k[lvl].device)
            buf[:, : lb[lvl].shape[1]] = lb[lvl]
            self._b.append(buf)

    @torch.no_grad()
    def _bias_level_update(self, level: int, p_lo: int, p_hi: int) -> None:
        """Recompute b for parents [p_lo, p_hi) at `level` from their current children (compose_level_bias restricted to
        a parent range)."""
        from recsa.level_bias import bucket_alpha, entropy
        c, g = self.c, self._g_b
        dev = self._b[level].device
        lo = p_lo * c
        hi = min(p_hi * c, self._sizes[level - 1])
        ck = self._k[level - 1][0, lo:hi]
        if ck.device != dev:
            ck = ck.to(dev, non_blocking=False)  # offload: <=c host leaf rows
        alpha = bucket_alpha(ck.float(), hi - lo, self._probe_w,
                             self._probe_tau, c)          # [P, c, H]
        h = entropy(alpha)                                # [P, H]
        if level >= 2:
            hp = self._b[level - 1][0, lo:hi] / g          # child H (final for closed children)
            pad = (p_hi - p_lo) * c - hp.shape[0]
            if pad:
                hp = torch.nn.functional.pad(hp, (0, 0, 0, pad))
            h = h + (alpha * hp.reshape(p_hi - p_lo, c, -1)).sum(1)
        self._b[level][0, p_lo:p_hi] = g * h

    @torch.no_grad()
    def _refresh_open_bias(self) -> None:
        """Update b after one appended token: fill newborn levels, then recompute the open parent chain bottom-up."""
        if self._g_b <= 0.0:
            return
        H_kv = self._k[0].shape[2]
        while len(self._b) < len(self._sizes):
            lvl = len(self._b)
            self._b.append(torch.zeros(
                1, self._bias_cap(lvl), H_kv, dtype=torch.float32,
                device=self._k[lvl].device))
            self._bias_level_update(lvl, 0, self._sizes[lvl])
        for level in range(1, len(self._sizes)):
            p = (self._sizes[level - 1] - 1) // self.c
            self._bias_level_update(level, p, p + 1)

    def layers_bias(self) -> Optional[List[Optional[torch.Tensor]]]:
        """Current per-level bias views for recsa_forward(level_bias=...): None when the clamped gate is 0 or absent,
        else [None, b_1, ...] fp32 [1, N_l, H]."""
        if self._g_b <= 0.0:
            return None
        return [None] + [self._b[lvl][:, : self._sizes[lvl]]
                         for lvl in range(1, len(self._sizes))]

    @torch.no_grad()
    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """Append one token [B, H, D] or a chunk [B, n, H, D] of new K/V."""
        if k_t.dim() == 3:
            k_t, v_t = k_t.unsqueeze(1), v_t.unsqueeze(1)
        if k_t.shape != v_t.shape:
            raise ValueError(f"k/v chunk shapes differ: {tuple(k_t.shape)} vs {tuple(v_t.shape)}")
        if k_t.dtype != self._k[0].dtype or v_t.dtype != self._v[0].dtype:
            raise ValueError(
                f"append dtype {k_t.dtype}/{v_t.dtype} != tree dtype "
                f"{self._k[0].dtype} — a silent cast would break builder provenance")
        n_new = k_t.shape[1]
        if self._sizes[0] + n_new > self._capacity:
            raise RuntimeError(
                f"reserve exhausted: {self._sizes[0]} + {n_new} > capacity "
                f"{self._capacity}; construct with a larger reserve"
            )
        for i in range(n_new):
            t = self._sizes[0]
            if self._offload:
                self._k[0][:, t].copy_(k_t[:, i])
                self._v[0][:, t].copy_(v_t[:, i])
                if self._mirror is not None and t >= self._mirror[2]:
                    # also into the device mirror (row t - R0)
                    self._mirror[0][:, t - self._mirror[2]].copy_(k_t[:, i])
                    self._mirror[1][:, t - self._mirror[2]].copy_(v_t[:, i])
            else:
                self._k[0][:, t] = k_t[:, i]
                self._v[0][:, t] = v_t[:, i]
            self._sizes[0] = t + 1
            self._refresh_open_parents()
            self._refresh_open_bias()

    # -- graph-safe decode ---------------------------------

    def bind_cursor(self, t_dev: torch.Tensor, window=None,
                    defer_v: bool = False, exact: bool = False,
                    fused: bool = False) -> None:
        """Attach the session cursor, an int64 [1] CUDA tensor holding t (tokens in context). append_cursor derives all
        step-varying geometry from it, so its launches can be replayed in a CUDA graph."""
        if not (t_dev.is_cuda and t_dev.dtype == torch.int64 and t_dev.numel() == 1):
            raise ValueError("cursor must be an int64 [1] CUDA tensor")
        if getattr(self, "_g_b", 0.0) > 0.0:
            raise NotImplementedError(
                "cursor (CUDA-graph) appends have no incremental b "
                "maintenance — run the eager session for a live gate_b")
        if self._k[0].shape[0] != 1:
            raise ValueError("cursor mode requires B == 1 (flat dim-1 offsets)")
        staged = False
        if self._probe_w is not None and self._mode in MLP_MODES:
            if not self._k[0].is_cuda:
                if not (self._offload and MLP_OFFLOAD_GRAPH == "on"):
                    raise ValueError(
                        "MLP-pooler cursor appends need resident (GPU) "
                        "leaves: offloaded children are pinned-host rows a "
                        "device-indexed torch gather cannot read, and the MLP "
                        "modes have no fused pool kernel — run the eager "
                        "session under offload"
                        + ("" if not self._offload else
                           " (RECSA_MLP_OFFLOAD_GRAPH=off)"))
                kd, vd = self._k[0].shape[3], self._v[0].shape[3]
                if any(x < 1 or x & (x - 1) for x in (self.c, kd, vd)):
                    # recsa_leaf_store_bucket_kernel tiles [c, D] blocks
                    raise ValueError(
                        "MLP offload cursor appends need a power-of-2 "
                        f"compression rate and head dims, got c={self.c}, "
                        f"K={kd}, V={vd}")
                staged = True
            params = [self._probe_w, self._probe_tau, *self._probe_extra.values()]
            if self._probe_gate is not None:
                params.append(self._probe_gate)
            if (self._k[0].dtype == torch.float64
                    or any(p.dtype != torch.float32 for p in params)):
                raise ValueError(
                    "MLP-pooler cursor appends need fp32 probe params and a "
                    "non-fp64 tree (the eager DecodeBucketPoolMLP dispatch)")
            if (window is None or window.t_dev is not t_dev
                    or (window.c, window.d, window.capacity)
                    != (self.c, self._k[0].shape[3], self._capacity)):
                raise ValueError(
                    "MLP-pooler cursor appends need a BucketCursor over this "
                    "cursor at this state's c / head dim / capacity")
        elif self._probe_w is not None and not (
                self._shared and FUSED_POOL_DEFAULT and self._fused_eligible()):
            raise ValueError(
                "probe-pooled cursor appends require the fused pool "
                "(shared_std + RECSA_FUSED_POOL=1 + eligible shapes); the "
                "torch bucket pool slices by host ints and cannot replay")
        if staged:
            st_shape = (1, self.c, self._k[0].shape[2])
            self._stage_k = torch.zeros(*st_shape, self._k[0].shape[3],
                                        dtype=self._k[0].dtype, device=self._dev)
            self._stage_v = torch.zeros(*st_shape, self._v[0].shape[3],
                                        dtype=self._v[0].dtype, device=self._dev)
            self._stage_idx = torch.arange(self.c, device=self._dev)
        else:
            self._stage_k = self._stage_v = self._stage_idx = None
        self._defer_v = False
        ov2 = self._offload and staged and OFFLOAD_POOL_V2 == "on"
        self._offload_v2 = False
        if defer_v:
            if self._probe_w is None or self._mode not in MLP_MODES:
                raise ValueError("defer_v is the MLP-pooler cursor pool")
            if not getattr(window, "fused", False):
                raise ValueError("defer_v needs a fused BucketCursor (idx_z windows)")
            if self._offload and not ov2:
                raise ValueError("defer_v needs resident leaves: the K-only "
                                 "pool would index the pinned-host level 0")
            self._offload_v2 = ov2
            self.check_spare_rows()
            self._tau_f = float(self._probe_tau)
            self._gate_f = 1.0 if self._probe_gate is None else float(self._probe_gate)
            self._defer_v = True
        self._exact = False
        if exact:
            if not self._defer_v:
                raise ValueError("exact pool segments build on the defer_v "
                                 "K-only cursor")
            from recsa.mlp_pool_exact import exact_eligible
            self._exact = (not self._offload or self._offload_v2) and exact_eligible(
                self._mode, self._k[0].shape[2], self._k[0].shape[3], self.c,
                self._dev if self._offload_v2 else self._k[0].device, self._k[0].dtype)
        self._fused = False
        self._fused_wt = None
        if fused:
            if not exact:
                raise ValueError("the fused pool builds on the exact "
                                 "segments")
            if self._exact:
                from recsa.mlp_pool_exact import fused_eligible, fused_weights
                self._fused = fused_eligible(self._probe_w, self._probe_extra,
                                             self._dev if self._offload_v2
                                             else self._k[0].device)
                if self._fused:
                    # the kernel's transposed copies of the frozen MLP weights
                    self._fused_wt = fused_weights(self._probe_extra)
        self._t_dev = t_dev
        self._cursor_win = window

    def spare_row(self, level: int) -> int:
        """The spare last row of level `level`'s growth buffer, past every row an append within the capacity writes."""
        cp = self.c ** level
        return (self._capacity + cp - 1) // cp

    def check_spare_rows(self) -> None:
        """Check that every level buffer's spare row exists, lies beyond the last row any append can write, and is +0.0
        (DecodeBucketPoolMLP.cursor_k reads pad rows from it)."""
        for lvl, buf in enumerate(self._k):
            r = self.spare_row(lvl)
            if not (buf.shape[1] > r > (self._capacity - 1) // (self.c ** lvl)):
                raise RuntimeError(
                    f"level {lvl}: growth buffer of {buf.shape[1]} rows has no "
                    f"spare row {r} beyond the capacity {self._capacity}")
            if bool(buf[:, r].reshape(-1).view(torch.uint8).any()):
                raise RuntimeError(
                    f"level {lvl}: spare row {r} is not +0.0 — the K pool of "
                    "the deferred-V cursor would gather garbage pad rows")

    def append_cursor(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """Graph-replayable single-token append: leaf index_copy_ at the cursor + one cursor-mode pool launch per level
        into the open parent slot."""
        if getattr(self, "_offload_v2", False):
            return self._append_cursor_offload_v2(k_t, v_t)
        t_dev = self._t_dev
        # the exact-order level-1 MLP pool writes the leaf row itself
        exact = getattr(self, "_exact", False) and len(self._sizes) > 1
        if exact:
            H_kv, Kd = self._k[0].shape[2], self._k[0].shape[3]
            if (k_t.dtype != self._k[0].dtype or v_t.dtype != self._v[0].dtype
                    or k_t.numel() != H_kv * Kd or v_t.numel() != H_kv * Kd):
                raise ValueError("append_cursor: leaf rows must be [1, 1, H_kv, D] "
                                 "in the tree dtype")
        elif self._offload:
            H_kv, Kd = self._k[0].shape[2], self._k[0].shape[3]
            Vd = self._v[0].shape[3]
            with torch.cuda.device(self._dev):
                if getattr(self, "_stage_k", None) is not None:
                    _leaf_store_bucket()[(H_kv,)](
                        self._k[0], self._v[0], self._stage_k, self._stage_v,
                        k_t.contiguous(), v_t.contiguous(), t_dev,
                        H_kv=H_kv, K=Kd, V=Vd, C=self.c)
                else:
                    _leaf_store()[(H_kv,)](
                        self._k[0], self._v[0],
                        k_t.contiguous(), v_t.contiguous(), t_dev,
                        H_kv=H_kv, K=Kd, V=Vd)
            if self._mirror is not None:
                self._mirror_store(k_t, v_t)  # the same row into the leaf mirror
        else:
            self._k[0].index_copy_(1, t_dev, k_t)
            self._v[0].index_copy_(1, t_dev, v_t)
        if self._probe_w is None:
            kb = _raw_build_kernel()
            H_kv, Kd = self._k[0].shape[2], self._k[0].shape[3]
            Vd = self._v[0].shape[3]
            with torch.cuda.device(self._dev):
                for level in range(1, len(self._sizes)):
                    kb[(1, H_kv)](
                        self._k[level - 1], self._v[level - 1],
                        self._k[level], self._v[level],
                        N_child=0, N_parent=1, B=1,  # cursor-derived in-kernel
                        H_kv=H_kv, K=Kd, V=Vd,
                        COMPRESSION_RATE=self.c, BLOCK_SIZE=8,
                        t_ptr=t_dev, CHILD_POWER=self.c ** (level - 1),
                        HAS_CURSOR=1)
            return
        if self._mode in MLP_MODES:
            pool = decode_bucket_pool_mlp(
                self._mode, self._k[0].shape[2], self._k[0].shape[3], self.c,
                self._dev)
            if exact:
                leaf = (self._v[0], k_t.contiguous(), v_t.contiguous(), t_dev)
                n_lv = len(self._sizes) - 1
                if getattr(self, "_fused", False):
                    # fused: one launch per FUSED_LEVELS_PER_LAUNCH levels
                    from recsa import mlp_pool_exact as _mpe
                    per = _mpe.FUSED_LEVELS_PER_LAUNCH
                    level = 1
                    while level <= n_lv:
                        two = per >= 2 and level < n_lv
                        pool.cursor_k_fused(
                            self._k[level - 1], self._cursor_win[level],
                            self._probe_w, self._tau_f, self._gate_f,
                            self._probe_extra, self._k[level],
                            leaf if level == 1 else None,
                            self._cursor_win[level + 1] if two else None,
                            self._k[level + 1] if two else None,
                            self._fused_wt)
                        level += 2 if two else 1
                    return
                for level in range(1, n_lv + 1):
                    pool.cursor_k_exact(self._k[level - 1], self._cursor_win[level],
                                        self._probe_w, self._tau_f, self._gate_f,
                                        self._probe_extra, self._k[level],
                                        leaf if level == 1 else None)
                return
            if self._defer_v:
                for level in range(1, len(self._sizes)):
                    pool.cursor_k(self._k[level - 1], self._cursor_win[level],
                                  self._probe_w, self._tau_f, self._gate_f,
                                  self._probe_extra, self._k[level])
                return
            for level in range(1, len(self._sizes)):
                child_k, child_v, win = self._cursor_children(level)
                pool.cursor(child_k, child_v, win, self._probe_w,
                            self._probe_tau, self._probe_gate,
                            self._probe_extra, self._k[level], self._v[level])
            return
        launch = _fused_pool_launch()
        with torch.cuda.device(self._dev):
            for level in range(1, len(self._sizes)):
                launch(self._k[level - 1], self._v[level - 1],
                       self._probe_w, self._probe_tau, self._probe_gate,
                       self._k[level], self._v[level], self.c,
                       t_dev=t_dev, child_power=self.c ** (level - 1))

    def _append_cursor_offload_v2(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """append_cursor of an offloaded MLP state with OFFLOAD_POOL_V2 "on": the write-through store and staging fill,
        then the K pool with level 1 reading the staging pair through _stage_window()."""
        t_dev = self._t_dev
        H_kv, Kd = self._k[0].shape[2], self._k[0].shape[3]
        Vd = self._v[0].shape[3]
        with torch.cuda.device(self._dev):
            _leaf_store_bucket()[(H_kv,)](
                self._k[0], self._v[0], self._stage_k, self._stage_v,
                k_t.contiguous(), v_t.contiguous(), t_dev,
                H_kv=H_kv, K=Kd, V=Vd, C=self.c)
        if self._mirror is not None:
            self._mirror_store(k_t, v_t)
        pool = decode_bucket_pool_mlp(self._mode, H_kv, Kd, self.c, self._dev)
        n_lv = len(self._sizes) - 1
        w1 = self._stage_window()

        def child(level):
            if level == 1:
                return self._stage_k, w1
            return self._k[level - 1], self._cursor_win[level]
        if self._exact:
            if self._fused:
                from recsa import mlp_pool_exact as _mpe
                per = _mpe.FUSED_LEVELS_PER_LAUNCH
                level = 1
                while level <= n_lv:
                    two = per >= 2 and level < n_lv
                    ck, win = child(level)
                    pool.cursor_k_fused(
                        ck, win, self._probe_w, self._tau_f, self._gate_f,
                        self._probe_extra, self._k[level], None,
                        self._cursor_win[level + 1] if two else None,
                        self._k[level + 1] if two else None, self._fused_wt)
                    level += 2 if two else 1
                return
            for level in range(1, n_lv + 1):
                ck, win = child(level)
                pool.cursor_k_exact(ck, win, self._probe_w, self._tau_f, self._gate_f,
                                    self._probe_extra, self._k[level], None)
            return
        for level in range(1, n_lv + 1):
            ck, win = child(level)
            pool.cursor_k(ck, win, self._probe_w, self._tau_f, self._gate_f,
                          self._probe_extra, self._k[level])

    def _stage_window(self):
        """The level-1 window over the staging pair: the session window with child rows idx / idx_z replaced by
        0..c-1."""
        return self._cursor_win[1]._replace(idx=self._stage_idx, idx_z=self._stage_idx)

    def _mirror_store(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        mk, mv, r0 = self._mirror
        H_kv, Kd = self._k[0].shape[2], self._k[0].shape[3]
        Vd = self._v[0].shape[3]
        with torch.cuda.device(self._dev):
            _mirror_store()[(H_kv,)](mk, mv, k_t.contiguous(), v_t.contiguous(),
                                     self._t_dev, r0, H_kv=H_kv, K=Kd, V=Vd)

    # -- device recency mirror of the offloaded leaves -----------------------

    def attach_leaf_mirror(self, r0: int) -> None:
        """Keep a device copy of this offloaded layer's leaf rows >= r0 (r0 a multiple of c) for the leaf gather to read
        (mirror row r == pool row r0 + r)."""
        if not self._offload:
            raise ValueError("a leaf mirror needs offloaded (pinned-host) leaves")
        if getattr(self, "_t_dev", None) is not None:
            raise RuntimeError("attach the leaf mirror before bind_cursor "
                               "(a CUDA graph captures its addresses)")
        if self._k[0].device.type == "cuda" or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("leaf mirror: host level 0 expected, no capture")
        r0 = int(r0)
        kh, vh = self._k[0], self._v[0]
        if r0 < 0 or r0 % self.c or r0 >= kh.shape[1]:
            raise ValueError(f"mirror start row {r0} must be a multiple of c={self.c} "
                             f"in [0, {kh.shape[1]})")
        self.detach_leaf_mirror()
        rows = kh.shape[1] - r0
        mk = torch.zeros(1, rows, kh.shape[2], kh.shape[3], dtype=kh.dtype, device=self._dev)
        mv = torch.zeros(1, rows, vh.shape[2], vh.shape[3], dtype=vh.dtype, device=self._dev)
        n = self._sizes[0]
        if n > r0:
            mk[:, : n - r0].copy_(kh[:, r0:n])
            mv[:, : n - r0].copy_(vh[:, r0:n])
        from recsa.parallel import register_leaf_mirror, unregister_leaf_mirror
        self._mirror = (mk, mv, r0)
        self._mirror_key = register_leaf_mirror(kh, self._mirror)
        weakref.finalize(kh, unregister_leaf_mirror, self._mirror_key)

    def detach_leaf_mirror(self) -> None:
        """Drop the leaf mirror (gathers read the pool again)."""
        if self._mirror is None:
            return
        if getattr(self, "_t_dev", None) is not None:
            raise RuntimeError("cannot detach a leaf mirror under a bound cursor")
        from recsa.parallel import unregister_leaf_mirror
        unregister_leaf_mirror(self._mirror_key)
        self._mirror = None

    def _cursor_children(self, level: int):
        """(child_k, child_v, window) the MLP cursor pool reads at `level`."""
        win = self._cursor_win[level]
        if level == 1 and getattr(self, "_stage_k", None) is not None:
            return self._stage_k, self._stage_v, win._replace(idx=self._stage_idx)
        return self._k[level - 1], self._v[level - 1], win

    def sync_sizes(self, t: int) -> None:
        """Restore the host mirror after cursor appends: sizes[l] = ceil(t / c^l)."""
        if compute_num_layers(t, self.c, self.mtn) != len(self._sizes):
            raise RuntimeError(
                f"sync_sizes({t}): level count would change; run the host "
                "(eager) append for the band-crossing step")
        for lvl in range(len(self._sizes)):
            self._sizes[lvl] = (t + self.c ** lvl - 1) // (self.c ** lvl)

    # -- internals ----------------------------------------------------------

    def _alloc_level(self, level: int, init_k: torch.Tensor, init_v: torch.Tensor,
                     buf=None) -> None:
        cap = (self._capacity + self.c ** level - 1) // (self.c ** level) + 1
        if buf is not None:
            # caller pre-allocated (see preallocate): copy in, allocate nothing
            buf_k, buf_v = buf
            if buf_k.shape[1] < init_k.shape[1] or buf_v.shape[1] < init_v.shape[1]:
                raise ValueError(
                    f"level {level}: preallocated width {buf_k.shape[1]} < "
                    f"{init_k.shape[1]} needed")
            buf_k[:, : init_k.shape[1]].copy_(init_k)
            buf_v[:, : init_v.shape[1]].copy_(init_v)
            self._k.append(buf_k)
            self._v.append(buf_v)
            self._sizes.append(init_k.shape[1])
            return

        buf_k = init_k.new_zeros(init_k.shape[0], cap, *init_k.shape[2:])
        buf_v = init_v.new_zeros(init_v.shape[0], cap, *init_v.shape[2:])
        buf_k[:, : init_k.shape[1]] = init_k
        buf_v[:, : init_v.shape[1]] = init_v
        self._k.append(buf_k)
        self._v.append(buf_v)
        self._sizes.append(init_k.shape[1])

    def _level_params(self, level: int):
        if self._probe_w is None:
            return None, None
        if self._shared or self._mode in MLP_MODES:
            return self._probe_w, self._probe_tau
        return self._probe_w[level - 1], self._probe_tau[level - 1]

    def _pool(self, level: int, child_k: torch.Tensor, child_v: torch.Tensor):
        if self._probe_w is None and child_k.is_cuda:
            w = child_k.shape[1]
            if w == 1:
                return child_k.clone(), child_v.clone()
            with torch.cuda.device(child_k.device):
                pk, pv = _kernel_build()(
                    child_k.contiguous(), child_v.contiguous(), self.c,
                    max(1, (w + self.c - 1) // self.c))
            return pk[1], pv[1]
        w_l, tau_l = self._level_params(level)
        if self._mode in MLP_MODES:
            if child_k.device != self._dev:
                child_k = child_k.to(self._dev)
                child_v = child_v.to(self._dev)
            px = self._probe_extra
            gate = self._probe_gate
            if (child_k.shape[0] == 1 and child_k.shape[1] <= self.c
                    and child_k.dtype != torch.float64
                    and w_l.dtype == torch.float32
                    and tau_l.dtype == torch.float32
                    and (gate is None or gate.dtype == torch.float32)
                    and all(t.dtype == torch.float32 for t in px.values())):
                return decode_bucket_pool_mlp(
                    self._mode, child_k.shape[2], child_k.shape[3], self.c,
                    child_k.device)(child_k, child_v, w_l, tau_l, gate, px)
            return _pool_one_level_mlp(
                child_k, child_v, self.c, self._mode, w_l, tau_l,
                gate, px)
        gate = self._probe_gate if self._shared else None
        if (self._shared and w_l is not None and child_k.shape[0] == 1
                and child_k.shape[1] <= self.c
                and child_k.dtype != torch.float64
                and w_l.dtype == torch.float32 and tau_l.dtype == torch.float32
                and (gate is None or gate.dtype == torch.float32)):
            return decode_bucket_pool(
                child_k.shape[2], child_k.shape[3], self.c, child_k.device
            )(child_k, child_v, w_l, tau_l, gate)
        return _pool_one_level(
            child_k, child_v, self.c, w_l, tau_l,
            standardize=self._shared, probe_gate=gate,
        )

    def _fused_eligible(self) -> bool:
        """Shape/dtype gate for the fused probe pool (cached; the env flag is checked separately so tests can toggle
        FUSED_POOL_DEFAULT at runtime)."""
        ok = self._fused_ok
        if ok is None:
            k0, v0 = self._k[0], self._v[0]
            kd, vd = k0.shape[3], v0.shape[3]
            floats = (torch.bfloat16, torch.float16, torch.float32)
            ok = self._fused_ok = bool(
                self._shared
                and self._probe_w is not None and self._probe_tau is not None
                and (k0.is_cuda or self._offload) and k0.shape[0] == 1
                and k0.dtype in floats and v0.dtype in floats
                and self._probe_w.dim() == 1
                and self._probe_w.dtype == torch.float32
                and self._probe_tau.dim() == 0
                and self._probe_tau.dtype == torch.float32
                and (self._probe_gate is None
                     or (self._probe_gate.dim() == 0
                         and self._probe_gate.dtype == torch.float32))
                and self.c >= 2 and (self.c & (self.c - 1)) == 0
                and kd >= 16 and (kd & (kd - 1)) == 0
                and vd >= 16 and (vd & (vd - 1)) == 0)
        return ok

    def _refresh_open_parents(self) -> None:
        need = compute_num_layers(self._sizes[0], self.c, self.mtn)
        if need > len(self._sizes):
            level = len(self._sizes)
            ck = self._k[level - 1][:, : self._sizes[level - 1]]
            cv = self._v[level - 1][:, : self._sizes[level - 1]]
            nk, nv = self._pool(level, ck, cv)
            self._alloc_level(level, nk, nv)
        if (self._probe_w is None and (self._k[0].is_cuda or self._offload)
                and self._k[0].shape[0] == 1):
            kb = _raw_build_kernel()
            H_kv, Kd = self._k[0].shape[2], self._k[0].shape[3]
            Vd = self._v[0].shape[3]
            with torch.cuda.device(self._dev):
                for level in range(1, len(self._sizes)):
                    n_child = self._sizes[level - 1]
                    p = (n_child - 1) // self.c
                    lo = p * self.c
                    kb[(1, H_kv)](
                        self._k[level - 1][:, lo:n_child],
                        self._v[level - 1][:, lo:n_child],
                        self._k[level][:, p:p + 1],
                        self._v[level][:, p:p + 1],
                        N_child=n_child - lo, N_parent=1, B=1,
                        H_kv=H_kv, K=Kd, V=Vd,
                        COMPRESSION_RATE=self.c, BLOCK_SIZE=8)
                    self._sizes[level] = p + 1
            return
        if self._probe_w is not None and FUSED_POOL_DEFAULT and self._fused_eligible():
            launch = _fused_pool_launch()
            with torch.cuda.device(self._dev):
                for level in range(1, len(self._sizes)):
                    n_child = self._sizes[level - 1]
                    p = (n_child - 1) // self.c
                    lo = p * self.c
                    launch(
                        self._k[level - 1][:, lo:n_child],
                        self._v[level - 1][:, lo:n_child],
                        self._probe_w, self._probe_tau, self._probe_gate,
                        self._k[level][:, p:p + 1],
                        self._v[level][:, p:p + 1],
                        self.c,
                    )
                    self._sizes[level] = p + 1
            return
        for level in range(1, len(self._sizes)):
            n_child = self._sizes[level - 1]
            p = (n_child - 1) // self.c
            lo = p * self.c
            pk, pv = self._pool(
                level,
                self._k[level - 1][:, lo:n_child],
                self._v[level - 1][:, lo:n_child],
            )
            self._k[level][:, p] = pk[:, 0]
            self._v[level][:, p] = pv[:, 0]
            self._sizes[level] = p + 1


_vbatch_kernels = None


def _vbatch_launch():
    global _vbatch_kernels
    if _vbatch_kernels is None:
        from recsa.parallel import recsa_vbatch_gather_kernel, recsa_vbatch_scatter_kernel
        _vbatch_kernels = (recsa_vbatch_gather_kernel, recsa_vbatch_scatter_kernel)
    return _vbatch_kernels


class OpenParentVBatch:
    """The V half of the MLP cursor pool, deferred to the end of a decode step and batched over every layer of a
    session."""

    def __init__(self, states, window) -> None:
        if not states:
            raise ValueError("OpenParentVBatch needs at least one state")
        st0 = states[0]
        for st in states:
            if not getattr(st, "_defer_v", False) or st._cursor_win is not window:
                raise ValueError("every state must be bound with defer_v=True "
                                 "to this BucketCursor")
            v0 = st._v[0]
            if ((st.c, st._capacity) != (st0.c, st0._capacity)
                    or v0.shape[2:] != st0._v[0].shape[2:]
                    or v0.dtype != st0._v[0].dtype or v0.device != st0._v[0].device):
                raise ValueError("OpenParentVBatch: states differ in c / capacity / "
                                 "head shape / dtype / device")
        self.states, self.win = list(states), window
        self.n_levels = 0
        self._tabs: list = []
        self._idx1 = None
        H, Vd = st0._v[0].shape[2], st0._v[0].shape[3]
        dev = st0._dev if getattr(st0, "_offload_v2", False) else st0._v[0].device
        self._stage = torch.empty(len(states), 1, st0.c, H, Vd,
                                  dtype=torch.float32, device=dev)
        self._vpar = torch.empty(len(states), 1, H, Vd, dtype=st0._v[0].dtype,
                                 device=dev)

    def prepare(self) -> None:
        nl = self.states[0].num_layers
        dev = self._stage.device
        if any(st.num_layers != nl for st in self.states):
            raise RuntimeError("OpenParentVBatch: states disagree on the level count")
        if dev.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("OpenParentVBatch.prepare inside CUDA-graph capture "
                               "(a level-count change must be warmed eagerly)")
        for st in self.states:
            st.check_spare_rows()
        ov2 = [bool(getattr(st, "_offload_v2", False)) for st in self.states]
        if any(ov2) and not all(ov2):
            raise ValueError("OpenParentVBatch: offloaded and resident states mixed")
        self._idx1 = self.states[0]._stage_idx if all(ov2) else None
        tabs: list = [None]
        for level in range(1, nl):
            child = ([st._stage_v for st in self.states] if (level == 1 and self._idx1 is not None)
                     else [st._v[level - 1] for st in self.states])
            par = [st._v[level] for st in self.states]
            for b in child + par:
                if b.shape[0] != 1 or not b.is_contiguous() or b.device != dev:
                    raise ValueError("OpenParentVBatch: growth buffers must be "
                                     "contiguous B=1 tensors on the batch device")
            if dev.type == "cuda":
                tabs.append((
                    torch.tensor([b.data_ptr() for b in child], dtype=torch.int64, device=dev),
                    torch.tensor([b.data_ptr() for b in par], dtype=torch.int64, device=dev),
                    child, par))
            else:
                tabs.append((None, None, child, par))
        self._tabs = tabs
        self.n_levels = nl

    def __call__(self) -> None:
        if self.n_levels != self.states[0].num_layers:
            self.prepare()
        stage, vpar = self._stage, self._vpar
        n_states, c = stage.shape[0], stage.shape[2]
        row = stage.shape[3] * stage.shape[4]
        blk = 1 << (row - 1).bit_length()
        for level in range(1, self.n_levels):
            win = self.win[level]
            src_t, dst_t, child, par = self._tabs[level]
            if src_t is not None:
                gather, scatter = _vbatch_launch()
                with torch.cuda.device(stage.device):
                    gather[(n_states, c)](src_t, child[0],
                                          win.idx if (level != 1 or self._idx1 is None)
                                          else self._idx1,
                                          win.invalid4.view(torch.uint8), stage,
                                          ROW=row, BLOCK=blk, C=c)
            else:  # CPU (tests): the per-layer operand, layer by layer
                for i, cv in enumerate(child):
                    stage[i:i + 1].copy_(cv[:, win.idx].unsqueeze(1).float()
                                         .masked_fill(win.invalid5, 0.0))
            torch.div(stage.sum(dim=2), win.cnt4, out=vpar)
            if dst_t is not None:
                with torch.cuda.device(stage.device):
                    scatter[(n_states,)](dst_t, vpar, win.p, ROW=row, BLOCK=blk)
            else:
                for i, pv in enumerate(par):
                    pv.index_copy_(1, win.p, vpar[i:i + 1])
