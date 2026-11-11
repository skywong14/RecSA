#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MODEL_DIR="${1:-${MODEL_DIR:?usage: bash setup/get_model.sh <MODEL_DIR>}}"
REPO_ID="Qwen/Qwen3-4B"
REV="1cfa9a7208912126459214e8b04321603b3df60c"
[ "${CONDA_DEFAULT_ENV:-}" = recsa ] || { echo "ERROR: conda activate recsa first" >&2; exit 2; }
mkdir -p "$MODEL_DIR"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd -P)"
env -u HF_HUB_OFFLINE HF_HUB_DISABLE_TELEMETRY=1 python - "$REPO_ID" "$REV" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, dst = sys.argv[1:4]
p = snapshot_download(repo_id=repo, revision=rev, local_dir=dst,
                      allow_patterns=["*.json", "*.safetensors", "merges.txt", "LICENSE", "README.md"])
print(f"downloaded {repo}@{rev} -> {p}")
PY
cp "$HERE/model_files/recsa_graft.json" "$HERE/model_files/ARM_IDENTITY.json" "$MODEL_DIR/"
python "$HERE/verify_model.py" "$MODEL_DIR"
echo "export MODEL_DIR=$MODEL_DIR"
