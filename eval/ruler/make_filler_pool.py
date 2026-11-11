#!/usr/bin/env python
"""Build the RULER haystack filler pool from public DCLM-Baseline."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=str(HERE / "filler_rows.json"))
    a = ap.parse_args()
    import zstandard
    spec = json.loads(Path(a.rows).read_text())
    h = hashlib.sha256()
    with open(a.shard, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != spec["file_sha256"]:
        sys.exit(f"{a.shard}: sha256 {h.hexdigest()} != {spec['file_sha256']}")
    keep = set()
    for lo, hi in spec["row_ranges"]:
        keep.update(range(lo, hi + 1))
    last = max(keep)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    th, n = hashlib.sha256(), 0
    with open(a.shard, "rb") as fh, open(out, "w", encoding="utf-8") as fo:
        rd = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for i, line in enumerate(rd):
            if i > last:
                break
            if i not in keep:
                continue
            text = json.loads(line)["text"]
            th.update((text + "\n").encode("utf-8"))
            fo.write(json.dumps({"text": text, "source_name": spec["source_name"],
                                 "source_document_id": f"{spec['source']}:{Path(spec['file']).name}:{i}"}) + "\n")
            n += 1
    if n != spec["rows"] or th.hexdigest() != spec["text_sha256"]:
        out.unlink()
        sys.exit(f"filler pool mismatch: {n} rows (want {spec['rows']}), text sha256 {th.hexdigest()} "
                 f"(want {spec['text_sha256']})")
    print(f"wrote {out}: {n} documents, text sha256 {th.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
