#!/usr/bin/env python
"""usage: python setup/verify_model.py <MODEL_DIR>"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTRA = ("recsa_graft.json", "ARM_IDENTITY.json")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    m = Path(sys.argv[1])
    ident = json.loads((HERE / "model_files" / "ARM_IDENTITY.json").read_text())
    want = {**ident["weights_sha256"], **ident["small_files_sha256"],
            **{n: sha256(HERE / "model_files" / n) for n in EXTRA}}
    bad = []
    for name, sha in want.items():
        p = m / name
        if not p.is_file():
            bad.append(f"{name}: missing")
            continue
        got = sha256(p)
        print(f"{'OK  ' if got == sha else 'FAIL'} {name} {got}", flush=True)
        if got != sha:
            bad.append(f"{name}: sha256 {got} != {sha}")
    cfg = json.loads((m / "config.json").read_text()) if (m / "config.json").is_file() else {}
    for k, v in ident["config_facts"].items():
        if k in ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
                 "max_position_embeddings", "rope_theta", "tie_word_embeddings", "vocab_size") and cfg.get(k) != v:
            bad.append(f"config.json {k}={cfg.get(k)!r} != {v!r}")
    if cfg.get("rope_scaling"):
        bad.append(f"config.json carries rope_scaling {cfg['rope_scaling']} (must be null: native RoPE)")
    if bad:
        print("MODEL_VERIFY FAIL:\n  " + "\n  ".join(bad), flush=True)
        return 1
    print(f"MODEL_VERIFY OK {m} ({len(want)} files, revision {ident['snapshot_revision']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
