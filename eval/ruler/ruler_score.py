#!/usr/bin/env python
"""RULER score (official metric) of eval output directories."""
import argparse
import collections
import glob
import json
import os
import sys


def load(cell):
    rows = {}
    files = sorted(glob.glob(os.path.join(cell, "s*", "per_sample.jsonl")))
    for f in files:
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("score") is not None:
                    rows[r["sample_id"]] = float(r["score"])
    return rows, len(files)


def task_avg(rows):
    by = collections.defaultdict(list)
    for sid, v in rows.items():
        by[sid.rsplit("-", 1)[1]].append(v)
    per = {t: 100.0 * sum(v) / len(v) for t, v in sorted(by.items())}
    return (sum(per.values()) / len(per) if per else None), per, {t: len(v) for t, v in sorted(by.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--ids-from", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    keep = set(load(a.ids_from)[0]) if a.ids_from else None
    out = {}
    for cell in a.cells:
        rows, n_files = load(cell)
        if keep is not None:
            rows = {k: v for k, v in rows.items() if k in keep}
        avg, per, n = task_avg(rows)
        out[cell] = dict(score=avg, rows=len(rows), shards=n_files, tasks=len(per), per_task=per, rows_per_task=n)
        print(f"{cell}: {'n/a' if avg is None else f'{avg:.1f}'}  ({len(rows)} rows, {len(per)} tasks, "
              f"{n_files} shards)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
