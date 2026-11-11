from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import torch


@dataclass(frozen=True)
class RecSAParams:
    compression_rate: int
    max_top_nodes: int
    top_k_per_layer: int
    rope_base: float = 10000.0
    rope_mode: str = "internal"
    scale: Optional[float] = None


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    return torch.cat([x_lo * cos - x_hi * sin, x_lo * sin + x_hi * cos], dim=-1)


def build_rope_cache(
    dim: int,
    cache_size: int,
    device: torch.device,
    rope_base: float,
    dtype: torch.dtype,
    rope_mode: str = "internal",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rope_mode == "none":
        shape = (cache_size, dim // 2)
        return torch.ones(shape, dtype=dtype, device=device), torch.zeros(shape, dtype=dtype, device=device)
    if rope_mode != "internal":
        raise ValueError(f"unsupported rope_mode {rope_mode!r}; expected 'internal' or 'none'")
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))
    positions = torch.arange(cache_size, dtype=dtype, device=device)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def build_mean_tree(
    k: torch.Tensor,
    v: torch.Tensor,
    compression_rate: int,
    max_top_nodes: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    layers_k = [k]
    layers_v = [v]
    cur_k = k
    cur_v = v
    cur_len = k.shape[1]
    while cur_len > max_top_nodes:
        chunks_k = []
        chunks_v = []
        for start in range(0, cur_len, compression_rate):
            end = min(start + compression_rate, cur_len)
            chunks_k.append(cur_k[:, start:end].mean(dim=1))
            chunks_v.append(cur_v[:, start:end].mean(dim=1))
        cur_k = torch.stack(chunks_k, dim=1)
        cur_v = torch.stack(chunks_v, dim=1)
        layers_k.append(cur_k)
        layers_v.append(cur_v)
        cur_len = cur_k.shape[1]
    return layers_k, layers_v


def init_prev_selected_parents(
    seq_len: int,
    batch: int,
    h_kv: int,
    top_k: int,
    compression_rate: int,
    num_layers: int,
    device: torch.device,
) -> torch.Tensor:
    top_layer_power = compression_rate ** (num_layers - 1)
    t_indices = torch.arange(seq_len, dtype=torch.int32, device=device)
    rightmost_indices = t_indices // top_layer_power
    num_virtual_parents = rightmost_indices // compression_rate + 1
    parent_candidates = torch.arange(top_k, dtype=torch.int32, device=device)[None, :].expand(seq_len, -1)
    valid_mask = parent_candidates < num_virtual_parents[:, None]
    prev = torch.where(valid_mask, parent_candidates, torch.full_like(parent_candidates, -1))
    return prev[None, :, None, :].expand(batch, seq_len, h_kv, top_k).contiguous()


def parents_from_naive_trace(trace: Mapping, batch: int, seq_len: int, h_kv: int, params: RecSAParams, device) -> Dict[int, torch.Tensor]:
    num_layers = int(trace["num_layers"])
    prev = init_prev_selected_parents(
        seq_len,
        batch,
        h_kv,
        params.top_k_per_layer,
        params.compression_rate,
        num_layers,
        device,
    )
    per_layer: Dict[int, torch.Tensor] = {}
    for layer_idx in range(num_layers - 1, -1, -1):
        per_layer[layer_idx] = prev.detach().clone()
        if layer_idx > 0:
            selected = trace["layers"][layer_idx].selected_parents
            if selected is None:
                raise ValueError(f"missing selected parents for non-bottom layer {layer_idx}")
            prev = selected.detach().clone()
    return per_layer


def expanded_candidates(
    prev_selected: torch.Tensor,
    query_index: int,
    layer_idx: int,
    compression_rate: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    valid_parents = prev_selected[prev_selected >= 0].to(torch.long)
    if valid_parents.numel() == 0:
        raise ValueError("candidate frontier has no valid parents")
    layer_power = compression_rate ** layer_idx
    rightmost_idx = query_index // layer_power
    rightmost_child_idx = rightmost_idx % compression_rate
    num_candidates = (int(valid_parents.numel()) - 1) * compression_rate + rightmost_child_idx + 1
    cand_pos = torch.arange(num_candidates, dtype=torch.long, device=device)
    parent_off = torch.div(cand_pos, compression_rate, rounding_mode="floor")
    child_slot = cand_pos - parent_off * compression_rate
    parent_idx = valid_parents[parent_off]
    node_indices = parent_idx * compression_rate + child_slot
    rope_pos_q = num_candidates - 1
    return node_indices, cand_pos, rope_pos_q


def _selected_mask(node_indices: torch.Tensor, selected_nodes: Optional[torch.Tensor]) -> torch.Tensor:
    if selected_nodes is None:
        return torch.zeros_like(node_indices, dtype=torch.bool)
    valid = selected_nodes[selected_nodes >= 0].to(torch.long)
    if valid.numel() == 0:
        return torch.zeros_like(node_indices, dtype=torch.bool)
    return (node_indices[:, None] == valid[None, :]).any(dim=1)


def fixed_selection_forward_from_parents(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    per_layer_parents: Mapping[int, torch.Tensor],
    params: RecSAParams,
    query_indices: Optional[torch.Tensor] = None,
    no_summaries: bool = False,
    summary_norm: Optional[dict] = None,
    layers_kv: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
    _mass_out: Optional[dict] = None,
) -> torch.Tensor:
    """Fixed-selection RecSA forward from given per-layer parents (fp32/fp64 reference, differentiable)."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be rank-4 tensors")
    sn = None
    if summary_norm is not None:
        from recsa.parallel import summary_norm_spec
        sn = summary_norm_spec(summary_norm)
        if no_summaries:
            raise ValueError("summary_norm with no_summaries: nothing to normalise")
    batch, seq_len, hq, dim = q.shape
    h_kv = k.shape[2]
    v_dim = v.shape[-1]
    if hq % h_kv != 0:
        raise ValueError("Hq must be divisible by Hkv")
    groups = hq // h_kv
    scale = params.scale if params.scale is not None else dim ** -0.5
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    device = q.device

    if layers_kv is None:
        layers_k, layers_v = build_mean_tree(k, v, params.compression_rate, params.max_top_nodes)
    else:
        layers_k, layers_v = list(layers_kv[0]), list(layers_kv[1])
    num_layers = len(layers_k)
    cos_cache, sin_cache = build_rope_cache(
        dim,
        params.max_top_nodes + 1024,
        device,
        params.rope_base,
        compute_dtype,
        params.rope_mode,
    )

    if query_indices is None:
        active_queries = torch.arange(seq_len, dtype=torch.long, device=device)
    else:
        active_queries = query_indices.to(device=device, dtype=torch.long)

    out = torch.empty(batch, int(active_queries.numel()), hq, v_dim, dtype=compute_dtype, device=device)
    for b in range(batch):
        for out_t, t_tensor in enumerate(active_queries):
            t = int(t_tensor.item())
            for kv_h in range(h_kv):
                h0 = kv_h * groups
                h1 = h0 + groups
                q_group = q[b, t, h0:h1].to(compute_dtype)
                all_scores = []
                all_values = []
                seg_level = []      # the level of each score segment
                n_inc = {}          # summaries per non-bottom level
                for layer_idx in range(num_layers - 1, -1, -1):
                    prev = per_layer_parents[layer_idx][b, t, kv_h].to(device)
                    node_indices, cand_pos, rope_pos_q = expanded_candidates(
                        prev, t, layer_idx, params.compression_rate, device
                    )
                    if layer_idx > 0 and no_summaries:
                        include_mask = torch.zeros_like(node_indices, dtype=torch.bool)
                    elif layer_idx > 0:
                        selected_next = per_layer_parents[layer_idx - 1][b, t, kv_h].to(device)
                        include_mask = ~_selected_mask(node_indices, selected_next)
                    else:
                        include_mask = torch.ones_like(node_indices, dtype=torch.bool)
                    if layer_idx > 0 and sn is not None:
                        n_inc[layer_idx] = int(include_mask.sum().item())
                    if not bool(include_mask.any().item()):
                        continue
                    node_indices = node_indices[include_mask]
                    cand_pos = cand_pos[include_mask]
                    k_nodes = layers_k[layer_idx][b, node_indices, kv_h].to(compute_dtype)
                    v_nodes = layers_v[layer_idx][b, node_indices, kv_h].to(compute_dtype)
                    k_rope = apply_rope(k_nodes, cos_cache[cand_pos], sin_cache[cand_pos])
                    q_rope = apply_rope(q_group, cos_cache[rope_pos_q], sin_cache[rope_pos_q]) * float(scale)
                    all_scores.append(q_rope @ k_rope.transpose(0, 1))
                    all_values.append(v_nodes)
                    seg_level.append(layer_idx)
                assert all_scores, f"empty attention frontier at b={b} t={t} kv_h={kv_h}"
                scores = torch.cat(all_scores, dim=1)
                values = torch.cat(all_values, dim=0)
                probs = torch.softmax(scores, dim=1)
                if sn is not None:
                    # mass_cap on the probability vector
                    is_sum = torch.cat([torch.full((x.shape[1],), lvl > 0, dtype=torch.bool, device=device)
                                        for x, lvl in zip(all_scores, seg_level)])
                    assert bool((~is_sum).any()), "no leaf column: the forced chain reaches a leaf"
                    m = probs[:, is_sum].sum(dim=1)
                    if _mass_out is not None:
                        for gi in range(groups):
                            _mass_out[(b, out_t, h0 + gi)] = float(m[gi])
                    cap = m > sn.tau
                    if sn.cap_nref is not None and not sum(n_inc.values()) > sn.cap_nref:
                        cap = torch.zeros_like(cap)
                    f_s = torch.where(cap, sn.tau / m, torch.ones_like(m))
                    f_l = torch.where(cap, (1.0 - sn.tau) / (1.0 - m), torch.ones_like(m))
                    probs = probs * torch.where(is_sum[None, :], f_s[:, None], f_l[:, None])
                out[b, out_t, h0:h1] = probs @ values
    return out.to(q.dtype)


def dense_degenerate_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rope_base: float = 10000.0,
    rope_mode: str = "internal",
    scale: Optional[float] = None,
) -> torch.Tensor:
    batch, seq_len, hq, dim = q.shape
    h_kv = k.shape[2]
    v_dim = v.shape[-1]
    if hq % h_kv != 0:
        raise ValueError("Hq must be divisible by Hkv")
    groups = hq // h_kv
    scale = scale if scale is not None else dim ** -0.5
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    cos_cache, sin_cache = build_rope_cache(dim, seq_len + 1, q.device, rope_base, compute_dtype, rope_mode)

    out = torch.empty(batch, seq_len, hq, v_dim, dtype=compute_dtype, device=q.device)
    for b in range(batch):
        for t in range(seq_len):
            positions = torch.arange(t + 1, device=q.device, dtype=torch.long)
            for kv_h in range(h_kv):
                h0 = kv_h * groups
                h1 = h0 + groups
                q_group = q[b, t, h0:h1].to(compute_dtype)
                k_nodes = k[b, : t + 1, kv_h].to(compute_dtype)
                v_nodes = v[b, : t + 1, kv_h].to(compute_dtype)
                q_rope = apply_rope(q_group, cos_cache[t], sin_cache[t]) * float(scale)
                k_rope = apply_rope(k_nodes, cos_cache[positions], sin_cache[positions])
                scores = q_rope @ k_rope.transpose(0, 1)
                out[b, t, h0:h1] = torch.softmax(scores, dim=1) @ v_nodes
    return out.to(q.dtype)
