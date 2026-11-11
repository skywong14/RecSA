#!/usr/bin/env python3
"""usage: python scripts/eval/ppl_tokens.py --docset eval/ppl/docset.json --parquet-dir $RECSA_DATA/pg19/data --tokenizer
$MODEL_DIR --out $RECSA_DATA/pg19_tokens [--workers 8]"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppl_docset as D  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docset", required=True)
    ap.add_argument("--parquet-dir", required=True, help="directory of the emozilla/pg19 parquet files (train + test)")
    ap.add_argument("--tokenizer", required=True, help="the Qwen3-4B model dir (setup/get_model.sh)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    docset = Path(a.docset)
    ds = json.loads(docset.read_text())
    man = {}
    for line in open(docset.parent / "MANIFEST.sha256"):
        h, p = line.split(None, 1)
        man[p.strip()] = h
    out = Path(a.out)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    D.TOKENIZER = a.tokenizer
    pq_dir = Path(a.parquet_dir)
    E_MAIN, E_2M = ds["rules"]["E_main"], ds["rules"]["E_2m"]
    test_first = ds["rules"]["test_first"]
    jobs, want = [], {}
    for set_name, E in (("main", E_MAIN), ("two_m", E_2M)):
        for b in ds[set_name]:
            jobs.append((str(pq_dir / b["file"]), int(b["row"]), None, b["book_id"]))
            want[b["book_id"]] = (b["text_sha256"], f"{b['book_id']}_E{E}.npy", E)
    rows = {r["book_id"]: r for r in ds["test"]["rows"]}
    for bid in ds["test"]["books"]:
        r = rows[bid]
        jobs.append((str(pq_dir / ds["test_parquet"]), int(r["row"]), None, bid))
        want[bid] = (r["text_sha256"], f"{bid}_first.npy", test_first)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        jobs = [(p, row, tmp, key) for p, row, _, key in jobs]
        with Pool(max(1, min(a.workers, len(jobs)))) as pool:
            res = pool.map(D._tokenize_book, jobs, chunksize=1)
        bad = []
        for key, n_tok, text_sha, _ in res:
            sha, name, E = want[key]
            if text_sha != sha:
                bad.append(f"{key}: text sha256 {text_sha} != docset {sha}")
                continue
            ids = np.load(os.path.join(tmp, f"{key}.npy"))
            p = out / "tokens" / name
            np.save(p, np.ascontiguousarray(ids[:E], dtype=np.uint32))
            got = D.sha256_file(p)
            if man.get(f"tokens/{name}") != got:
                bad.append(f"tokens/{name}: sha256 {got} != MANIFEST {man.get(f'tokens/{name}')}")
    if bad:
        print("PPL TOKENS FAIL:\n  " + "\n  ".join(bad), flush=True)
        return 1
    shutil.copyfile(docset, out / "docset.json")
    shutil.copyfile(docset.parent / "MANIFEST.sha256", out / "MANIFEST.sha256")
    print(f"PPL TOKENS OK: {len(res)} token files in {out / 'tokens'} match MANIFEST.sha256", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
