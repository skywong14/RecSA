#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
TIER=default
if [[ "${1:-}" == "--full" ]]; then TIER=full; shift; fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "recsa" ]]; then
  echo "run_tests.sh: activate the 'recsa' conda env first (conda env create -f environment.yml)" >&2
  exit 2
fi
GPU_TAG="${CUDA_VISIBLE_DEVICES:-all}"; GPU_TAG="${GPU_TAG//,/_}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO/.triton_cache_tests/$(hostname)_g${GPU_TAG}}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TRITON_CACHE_DIR"
echo "run_tests.sh: tier=$TIER TRITON_CACHE_DIR=$TRITON_CACHE_DIR" >&2
if [[ $TIER == full ]]; then
  python -m pytest tests/ -q --durations=25 "$@"
else
  python -m pytest tests/ -q -m "not full" --durations=25 "$@"
fi
