#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TP=$REPO/third_party
DATA=${RECSA_DATA:-$TP/data}
HF=${HF_ENDPOINT:-https://huggingface.co}
mkdir -p "$DATA"

sha_ok() {  # sha_ok <file> <sha256>
  local got; got=$(sha256sum "$1" | cut -d' ' -f1)
  [[ $got == "$2" ]] || { echo "fetch.sh: $1: sha256 $got != $2" >&2; exit 1; }
}
get() {     # get <url> <dest> [sha256]
  local url=$1 dst=$2
  mkdir -p "$(dirname "$dst")"
  if [[ ! -s $dst ]]; then
    echo "fetch.sh: $url" >&2
    curl -fL --retry 3 -o "$dst.part" "$url" && mv "$dst.part" "$dst"
  fi
  [[ -z ${3:-} ]] || sha_ok "$dst" "$3"
}
tree_sha() {  # tree_sha <dir>: sha256 over the sorted `sha256sum` lines of every regular file (caches excluded)
  (cd "$1" && find . -type f ! -name UPSTREAM_COMMIT.txt ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z \
     | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}
github_tree() {  # github_tree <owner/repo> <commit> <dest dir> <tree sha256>
  local repo=$1 rev=$2 dst=$3 want=$4
  if [[ ! -d $dst ]]; then
    local tmp; tmp=$(mktemp -d)
    curl -fL --retry 3 "https://codeload.github.com/$repo/tar.gz/$rev" | tar -xz -C "$tmp"
    mv "$tmp"/*/ "$dst"
    rmdir "$tmp"
    echo "$rev" > "$dst/UPSTREAM_COMMIT.txt"
  else
    echo "fetch.sh: $dst exists" >&2
  fi
  local got; got=$(tree_sha "$dst")
  [[ $got == "$want" ]] || { echo "fetch.sh: $dst: tree sha256 $got != $want" >&2; exit 1; }
}

fetch_ruler() {
  local d=$DATA/ruler_sources
  get https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json "$d/squad_dev_v1.1.json" \
      95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9
  get "$HF/datasets/hotpotqa/hotpot_qa/resolve/1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet" \
      "$d/hotpot_validation.parquet" c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6
  if [[ ! -s $d/hotpot_dev_distractor_v1.json ]]; then
    python - "$d/hotpot_validation.parquet" "$d/hotpot_dev_distractor_v1.json" <<'PY'
import json, sys
import pyarrow.parquet as pq
rows = pq.read_table(sys.argv[1]).to_pylist()
out = [{"_id": r["id"], "question": r["question"], "answer": r["answer"],
        "context": [[t, list(s)] for t, s in zip(r["context"]["title"], r["context"]["sentences"])],
        "supporting_facts": [[t, i] for t, i in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"])],
        "type": r["type"], "level": r["level"]} for r in rows]
open(sys.argv[2], "w").write(json.dumps(out))
PY
  fi
  sha_ok "$d/hotpot_dev_distractor_v1.json" 9e46b0f534cf3cb9f7ceb7d8b995e3276800b1cc7d1784396c5a54fa047cd35b
  get "$HF/datasets/mlfoundations/dclm-baseline-1.0/resolve/a3b142c183aebe5af344955ae20836eb34dcf69b/global-shard_01_of_10/local-shard_2_of_10/shard_00000000_processed.jsonl.zst" \
      "$DATA/dclm/shard_00000000_processed.jsonl.zst" 3e6bd02d6321b8914991326ad0ecad61eb12fa577cb3fd9a0a69f89b4743c764
  [[ -s $DATA/ruler_filler/source_docs.jsonl ]] || python "$REPO/eval/ruler/make_filler_pool.py" \
      --shard "$DATA/dclm/shard_00000000_processed.jsonl.zst" --out "$DATA/ruler_filler/source_docs.jsonl"
}

fetch_pg19() {
  local d=$DATA/pg19/data rev=c021754c8e01c5b1cc83a1f549c1f97fbbb756b8
  local files f s
  files=$(python -c "import json, sys; d = json.load(open(sys.argv[1])); print(' '.join(sorted({b['file'] for b in d['main'] + d['two_m']} | {d['test_parquet']})))" \
          "$REPO/eval/ppl/docset.json")
  declare -A sha
  while read -r s f; do sha[$f]=$s; done <<'SHA'
9aae5ddf035760257458cff08d2575d78a15f84eff867af7a87eff0681b01bfc test-00000-of-00001-29a571947c0b5ccc.parquet
589252c8a3e870278db2b5d3f0da73e4845085e852cb024056c37b76d088ba41 train-00000-of-00023-5263dc4323e881d2.parquet
2e1872733cd492551e96a8faafc4493b432157d3e319ba44105f7ed0a90ba78d train-00001-of-00023-2b13c68dbeba63b5.parquet
047e8d8d9609a57bff14be4091d9226cebe5fab59f504eb3e5a917c8f69df5ac train-00005-of-00023-87dece6a0ab708e8.parquet
ed08834a0a5d4b63af11f94595561fb6988693ff3a75eacbf6e1a934cec39f78 train-00010-of-00023-dd5d8583a9f6a549.parquet
f19395403425b8650adab39208cdce17b1c8e1446aa8cd5090cc10fbccc0ed81 train-00011-of-00023-6c0bc91e6df2e661.parquet
750ebe5956aa6d52c1fb634352c6d3c5da19aec4b4ab30b3a4666970c05c24e2 train-00016-of-00023-8269d10108995a28.parquet
61c4add09ed3883b494e9bc2ec8b4eb01d6f8b190c7cd1ed248bbef74c6dcaa0 train-00017-of-00023-e240b232c0daddca.parquet
SHA
  for f in $files; do
    [[ -n ${sha[$f]:-} ]] || { echo "fetch.sh: pg19: no sha256 recorded for $f" >&2; exit 1; }
    get "$HF/datasets/emozilla/pg19/resolve/$rev/data/$f" "$d/$f" "${sha[$f]}"
  done
  : "${MODEL_DIR:?pg19: set MODEL_DIR (setup/get_model.sh) for the tokenizer}"
  python "$REPO/scripts/eval/ppl_tokens.py" --docset "$REPO/eval/ppl/docset.json" --parquet-dir "$d" \
      --tokenizer "$MODEL_DIR" --out "$DATA/pg19_tokens"
}

[[ "${CONDA_DEFAULT_ENV:-}" == "recsa" ]] || { echo "fetch.sh: activate the 'recsa' conda env first" >&2; exit 2; }
[[ $# -gt 0 ]] || set -- all
for what in "$@"; do
  case $what in
    all) fetch_ruler; fetch_pg19 ;;
    ruler) fetch_ruler ;;
    pg19) fetch_pg19 ;;
    *) echo "fetch.sh: unknown target $what" >&2; exit 2 ;;
  esac
done
echo "fetch.sh: done ($*); RECSA_DATA=$DATA" >&2
