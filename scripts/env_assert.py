#!/usr/bin/env python3
"""Resolve rope_theta from the model's config.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AUTO = "auto"
NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"


class RopeThetaMismatch(ValueError):
    """An explicit theta (config / flag) disagrees with the model's own rope_theta."""


def _theta_from_mapping(m: dict, where: str):
    rp = m.get("rope_parameters")
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        return float(rp["rope_theta"]), f"{where}:rope_parameters.rope_theta"
    if m.get("rope_theta") is not None:
        return float(m["rope_theta"]), f"{where}:rope_theta"
    rs = m.get("rope_scaling")
    if isinstance(rs, dict) and rs.get("rope_theta") is not None:
        return float(rs["rope_theta"]), f"{where}:rope_scaling.rope_theta"
    return None


def resolve_rope_theta(config_or_model_dir) -> tuple[float, str]:
    x = config_or_model_dir
    if isinstance(x, (str, Path)):
        p = Path(x)
        cfg_path = p / "config.json" if p.is_dir() else p
        if not cfg_path.is_file():
            raise ValueError(f"{x}: no config.json to resolve rope_theta from")
        with open(cfg_path) as f:
            got = _theta_from_mapping(json.load(f), str(cfg_path))
    elif isinstance(x, dict):
        got = _theta_from_mapping(x, "config")
    else:
        got = _theta_from_mapping({k: getattr(x, k, None) for k in ("rope_parameters", "rope_theta", "rope_scaling")},
                                  type(x).__name__)
    if got is None:
        raise ValueError(f"{x}: config states no rope_theta (rope_parameters / rope_theta / rope_scaling); "
                         "the Qwen3 default 10,000 is never assumed")
    return got


def is_auto(v) -> bool:
    return isinstance(v, str) and v.strip().lower() == AUTO


def parse_theta_arg(s):
    """argparse type: 'auto' (the model's value) or a positive number."""
    if is_auto(s):
        return AUTO
    try:
        f = float(s)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected 'auto' or a number, got {s!r}") from None
    if not f > 0:
        raise argparse.ArgumentTypeError(f"rope_theta must be > 0, got {s!r}")
    return f


def check_rope_theta(config_or_model_dir, expected=AUTO, what: str = "rope_theta") -> dict:
    """Resolve the model's rope_theta and check it against `expected` ('auto' adopts the model's value)."""
    theta, how = resolve_rope_theta(config_or_model_dir)
    if not is_auto(expected):
        exp = float(expected)
        if theta != exp:
            raise RopeThetaMismatch(
                f"{what} = {exp:g}, but the model resolves rope_theta = {theta:g} (via {how}); rope_theta is a "
                f"model property: use 'auto' or the model's own value")
    return {"rope_theta": theta, "rope_theta_resolution": how,
            "rope_theta_expected": AUTO if is_auto(expected) else float(expected)}


def graft_rope_base(graft: str, config_or_model_dir) -> tuple[str, float, dict]:
    mode, _, theta = graft.partition(":")
    rec = check_rope_theta(config_or_model_dir, parse_theta_arg(theta) if theta else AUTO, what=f"--graft {graft}")
    rec.update(graft=graft, graft_mode=mode)
    return mode, rec["rope_theta"], rec


def hybrid_thinking_template(tokenizer_or_template) -> bool:
    t = tokenizer_or_template
    if not isinstance(t, (str, dict)):
        t = getattr(t, "chat_template", None)
    if isinstance(t, dict):
        t = " ".join(str(v) for v in t.values())
    return bool(t) and "enable_thinking" in t


def chat_template_kwargs(tokenizer) -> dict:
    return {"enable_thinking": False} if hybrid_thinking_template(tokenizer) else {}


def resolved_theta_from_config_dir(model_dir: str | Path) -> tuple[float, str]:
    """(rope_theta as this environment's rotary embedding computes it, resolution method)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(model_dir))
    try:
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3RotaryEmbedding)
        inv = Qwen3RotaryEmbedding(cfg).inv_freq
        i = inv.shape[0] - 1
        dim = 2 * inv.shape[0]
        return float(inv[i] ** (-dim / (2.0 * i))), "inv_freq"
    except Exception:
        pass
    try:
        return resolve_rope_theta(cfg)[0], "config_attr"
    except ValueError:
        return 0.0, "config_attr"


def provenance() -> dict:
    import torch
    import transformers
    return {
        "python": sys.version.split()[0],
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }


def assert_recsa_env(model_dir: str | Path, expected_theta=AUTO) -> dict:
    """Check the config's rope_theta against `expected_theta` and against this environment's rotary embedding."""
    rec = check_rope_theta(model_dir, expected_theta, what=f"{model_dir}: expected rope_theta")
    theta = rec["rope_theta"]
    res, how = resolved_theta_from_config_dir(model_dir)
    if abs(res - theta) > 1e-3 * theta:
        raise RopeThetaMismatch(
            f"{model_dir}: this environment resolves rope_theta={res} (via {how}), the config states {theta} "
            f"({rec['rope_theta_resolution']}). This transformers cannot read the export's rope_parameters — "
            f"use the recsa env.")
    out = provenance()
    out.update(rec)
    out["rope_theta_forward"] = res
    out["rope_theta_forward_resolution"] = how
    out["model_dir"] = str(model_dir)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF export directory")
    ap.add_argument("--expected-theta", type=parse_theta_arg, default=AUTO,
                    help="'auto' (the model's own config value, default) or a number the model must resolve to")
    args = ap.parse_args()
    info = assert_recsa_env(args.model, args.expected_theta)
    print("ENV-OK " + json.dumps(info))


if __name__ == "__main__":
    main()
