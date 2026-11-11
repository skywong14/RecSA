#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${1:?usage: bash eval/ruler/gen_suites.sh <OUT> [L ...]}; shift
[[ $# -gt 0 ]] || set -- 32768 131072 262144 524288 1000000 2000000
: "${MODEL_DIR:?set MODEL_DIR (setup/get_model.sh)}"
DATA=${RECSA_DATA:-$REPO/third_party/data}
WORKERS=${WORKERS:-32}
cd "$REPO"
mkdir -p "$OUT"
gen() {  # gen <raw dir> <L> <samples per task> <seed family> [tasks ...]
  local raw=$1 L=$2 n=$3 fam=$4; shift 4
  local fr; fr=$(python -c "print(($L - 512) / $L)")
  local tasks=(); [[ $# -gt 0 ]] && tasks=(--tasks "$@")
  for s in $fam $((fam + 10)) $((fam + 20)); do
    rm -rf "$raw"
    python -m recsa.ruler.ruler_std --output-dir "$raw" --lengths "$L" --samples-per-task "$n" --seed "$s" \
      --target-tokenizer "$MODEL_DIR" --token-fraction "$fr" --workers "$WORKERS" \
      --filler-pool "$DATA/ruler_filler/source_docs.jsonl" --qa-sources "$DATA/ruler_sources" "${tasks[@]}" && return 0
    echo "gen_suites.sh: L=$L seed $s failed the length check; redrawing" >&2
  done
  return 1
}
render() {  # render <raw dir> <chat dir> <L> <rows per shard>
  python eval/ruler/render_chat_q3small.py --raw "$1/samples/eval_$3.jsonl" --tokenizer "$MODEL_DIR" --out "$2" \
    --L "$3" --per-shard "$4"
}
for L in "$@"; do
  case $L in
    2000000)
      gen "$OUT/raw_L2000000" 2000000 16 20262000 niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 \
        niah_multiquery vt fwe
      render "$OUT/raw_L2000000" "$OUT/chat" 2000000 1
      gen "$OUT/raw_L2000000_b" 2000000 16 20262000 niah_multikey_2 niah_multikey_3 niah_multivalue cwe qa_1 qa_2
      render "$OUT/raw_L2000000_b" "$OUT/chat_2m_b" 2000000 1 ;;
    32768) gen "$OUT/raw_L$L" "$L" 56 20261001; render "$OUT/raw_L$L" "$OUT/chat" "$L" 26 ;;
    *)     gen "$OUT/raw_L$L" "$L" 56 20261001; render "$OUT/raw_L$L" "$OUT/chat" "$L" 13 ;;
  esac
  echo "gen_suites.sh: L=$L done" >&2
done
