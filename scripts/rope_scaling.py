"""Inference-time YaRN for graft mode "external", shared by the evaluation drivers."""
from __future__ import annotations

import json
import math

import torch

ORIG_MAX = 32768


def yarn_factor(f: float, orig: int = ORIG_MAX) -> dict:
    f = float(f)
    if not f > 1.0:
        raise ValueError(f"YaRN factor must be > 1, got {f}")
    return {"rope_type": "yarn", "factor": f, "original_max_position_embeddings": int(orig)}


def yarn_for(L: int, orig: int = ORIG_MAX) -> dict | None:
    """None when L <= orig, else the smallest power of two f with orig * f >= L."""
    if L <= orig:
        return None
    f = 1
    while orig * f < L:
        f *= 2
    return yarn_factor(f, orig)


def parse_rope_scaling(s) -> dict | None:
    """A --rope-scaling-json value (JSON object with a rope_type) canonicalised with sorted keys; None for None."""
    if s is None:
        return None
    d = json.loads(s) if isinstance(s, str) else dict(s)
    if not isinstance(d, dict) or not d.get("rope_type"):
        raise ValueError("rope scaling must be a JSON object with a rope_type")
    return dict(sorted(d.items()))


def merge_rope_scaling(cfg, override: dict | None) -> dict | None:
    """Merge the override into config.rope_parameters (keeping rope_theta), then raise max_position_embeddings to
    original * factor."""
    if override is None:
        return None
    existing = dict(getattr(cfg, "rope_parameters", None) or {})
    merged = {**existing, **override}
    if merged.get("rope_theta") is None:
        theta = existing.get("rope_theta", getattr(cfg, "rope_theta", None))
        if theta is not None:
            merged["rope_theta"] = theta
    cfg.rope_scaling = merged
    factor = float(override.get("factor", 1.0))
    omp = int(override.get("original_max_position_embeddings", cfg.max_position_embeddings))
    cfg.max_position_embeddings = max(cfg.max_position_embeddings, int(omp * factor))
    return merged


def expected_attention_scaling(rs: dict | None) -> float:
    return 1.0 if rs is None else 0.1 * math.log(float(rs["factor"])) + 1.0


def check_rotary(model, rs: dict | None, theta: float | None = None) -> dict:
    """Check that the built model's rotary embedding matches `rs`; return a record of it."""
    rot = model.model.rotary_emb
    if rot.inv_freq.dtype != torch.float32:
        raise RuntimeError(f"rotary inv_freq is {rot.inv_freq.dtype}, not float32: true-position RoPE would carry a "
                           "position-proportional phase error (do not cast buffers to bf16)")
    want_type = "default" if rs is None else rs["rope_type"]
    want_scale = expected_attention_scaling(rs) if (rs is None or rs["rope_type"] == "yarn") else None
    got_scale = float(rot.attention_scaling)
    if rot.rope_type != want_type or (want_scale is not None and abs(got_scale - want_scale) > 1e-6):
        raise RuntimeError(f"rotary {rot.rope_type} x{got_scale}, want {want_type} x{want_scale} (rope_scaling {rs})")
    if theta is not None:
        rp = getattr(model.config, "rope_parameters", None) or {}
        got_theta = float(rp.get("rope_theta", getattr(model.config, "rope_theta", float("nan"))))
        if got_theta != float(theta):
            raise RuntimeError(f"rope_theta changed by the rope scaling merge: {got_theta} != {theta}")
    return dict(rope_scaling=rs, rotary_type=rot.rope_type, rotary_attention_scaling=got_scale,
                yarn_factor=(None if rs is None else float(rs.get("factor", 1.0))),
                inv_freq_dtype=str(rot.inv_freq.dtype).replace("torch.", ""))
