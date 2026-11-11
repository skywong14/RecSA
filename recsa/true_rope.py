"""True-position RoPE for the graft mode "external" (the global-RoPE ablation)."""
from __future__ import annotations

import torch

GRAFT_MODES = ("internal", "external")
DEFAULT_GRAFT_MODE = "internal"


def check_graft_mode(mode, where: str = "graft_mode") -> str:
    """Return mode if it is 'internal' or 'external', else raise ValueError."""
    if not isinstance(mode, str) or mode not in GRAFT_MODES:
        raise ValueError(f"{where}: graft mode must be one of {list(GRAFT_MODES)}, got {mode!r}")
    return mode


def rope_cos_sin(rotary_emb, like: torch.Tensor, positions: torch.Tensor):
    """(cos, sin) [1, T, D] in like.dtype for int positions [1, T] (or [T]) on like.device."""
    if positions.dim() == 1:
        positions = positions.unsqueeze(0)
    return rotary_emb(like, positions.to(like.device))


def positions_from(start: int, length: int, device) -> torch.Tensor:
    """int64 [1, length] = start .. start + length - 1."""
    return torch.arange(int(start), int(start) + int(length), dtype=torch.long, device=device).unsqueeze(0)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, T, H, D] rotated at the positions cos/sin [B, T, D] were built for."""
    from transformers.models.qwen3.modeling_qwen3 import rotate_half
    if cos.shape[-2] != x.shape[1] or cos.shape[-1] != x.shape[-1]:
        raise ValueError(f"rope cos {tuple(cos.shape)} does not match x {tuple(x.shape)} ([B, T, H, D])")
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)
