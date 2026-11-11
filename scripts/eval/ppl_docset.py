#!/usr/bin/env python3
"""PG-19 document set for the long-context perplexity evaluation."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PG19_PARQUET = os.path.join(os.environ.get("RECSA_DATA", "third_party/data"), "pg19", "data")   # emozilla/pg19 parquet
TOKENIZER = os.environ.get("MODEL_DIR", "")                  # setup/get_model.sh: Qwen/Qwen3-4B @1cfa9a72
D_CONFIG = os.environ.get("PPL_DECONTAM_CONFIG", "")
LEDGER = os.environ.get("PPL_DECONTAM_LEDGER", "")           # optional source-consumption ledger (jsonl)
SEED = "ppl-pg19-20261010"
N_MAIN = 10
E_MAIN = 1_000_000
E_2M = 2_000_000
TEST_FIRST = 131_072
TEST_MIN = 8_192
CHAR_PREFILTER = 2_500_000
NGRAM = 50
STRIDE = 256
MIN_DISTINCT = 16
BYTE_NS = (50, 13)
BYTE_RULE_N = 50
ANCHOR_MOD = 8
WIDELY_QUOTED_ROWS13 = 100
REFERENCE_RE = re.compile(r"dictionar|cyclop|encyclop|circle of knowledge|review digest|english biography|"
                          r"world of words|hobson-jobson|household manual|quotations", re.I)
GROUPS = [("bible", re.compile(r"\bbible\b", re.I)),
          ("shakespeare", re.compile(r"shakespeare", re.I)),
          ("debates_of_congress", re.compile(r"debates of congress", re.I))]
PG19_ID_RE = re.compile(rb"opensource_pg19_20260811/parquet/data/\*\.parquet:(train-\d{5}-of-\d{5}-[0-9a-f]+\.parquet)"
                        rb":(\d+)")
URL_RE = re.compile(rb"gutenberg\.org/ebooks/(\d+)")
SHA_RE = re.compile(rb'"source_sha256":\s*"sha256:([0-9a-f]{64})"')
IDS_RE = re.compile(rb'"input_ids":\s*\[')


def sha256_file(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def book_id(file: str, row: int) -> str:
    """Stable short id of a train-split book: pg19_<shard>_<row> (shard = the 5-digit parquet index)."""
    return f"pg19_{file.split('-')[1]}_{int(row)}"


def norm_title(t: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", (t or "").lower()).strip()


def gutenberg_no(url: str) -> str | None:
    m = re.search(r"ebooks/(\d+)", url or "")
    return m.group(1) if m else None


def rank_key(url: str) -> str:
    return hashlib.sha256(f"{SEED}:{url}".encode()).hexdigest()


def group_of(title: str, bid: str) -> str:
    for name, rx in GROUPS:
        if rx.search(title or ""):
            return name
    return bid


def ngram_weights() -> np.ndarray:
    """Fixed random odd uint64 weights: hash(w) = sum_i w_i * W_i (mod 2^64)."""
    rng = np.random.default_rng(20261010)
    return (rng.integers(0, 2 ** 63, size=NGRAM, dtype=np.uint64) * np.uint64(2) + np.uint64(1))


def ngram_hashes(ids: np.ndarray, W: np.ndarray) -> np.ndarray:
    """uint64 hash of every NGRAM-window of ids (len(ids) - NGRAM + 1 values)."""
    if ids.shape[0] < NGRAM:
        return np.zeros(0, dtype=np.uint64)
    win = np.lib.stride_tricks.sliding_window_view(ids.astype(np.uint64), NGRAM)
    return win @ W


def sampled_ngrams(ids: np.ndarray, W: np.ndarray):
    """(positions, hashes) of the NGRAM-windows starting every STRIDE tokens with >= MIN_DISTINCT distinct ids."""
    pos, hs = [], []
    for p in range(0, ids.shape[0] - NGRAM + 1, STRIDE):
        w = ids[p:p + NGRAM]
        if np.unique(w).shape[0] >= MIN_DISTINCT:
            pos.append(p)
            hs.append(int((w.astype(np.uint64) @ W)))
    return np.asarray(pos, dtype=np.int64), np.asarray(hs, dtype=np.uint64)


def anchored_hashes(words: list[str], n: int) -> np.ndarray:
    """uint64 blake2b-64 hashes of the n-word windows whose first word w has crc32(w) % ANCHOR_MOD == 0 (the same
    anchor rule on the book and the row side)."""
    out = []
    for i in range(0, len(words) - n + 1):
        if zlib.crc32(words[i].encode("utf-8", "surrogatepass")) % ANCHOR_MOD:
            continue
        out.append(int.from_bytes(hashlib.blake2b(" ".join(words[i:i + n]).encode("utf-8", "surrogatepass"),
                                                  digest_size=8).digest(), "little"))
    return np.asarray(out, dtype=np.uint64)


def window_bounds(E: int, L: int) -> tuple[int, int]:
    """The nested window of length L ending at E: tokens [E-L, E)."""
    if not 0 < L <= E:
        raise ValueError(f"window L={L} must be in (0, E={E}]")
    return E - L, E


def index_pool(parquet_dir: str) -> list[dict]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
        t = pq.read_table(f, columns=["short_book_title", "publication_date", "url", "text"])
        nch = pc.utf8_length(t["text"]).to_pylist()
        for i, (ti, pd, u, c) in enumerate(zip(t["short_book_title"].to_pylist(), t["publication_date"].to_pylist(),
                                               t["url"].to_pylist(), nch)):
            rows.append(dict(file=os.path.basename(f), row=i, book_id=book_id(os.path.basename(f), i), title=ti,
                             pub=pd, url=u, n_chars=int(c)))
    return rows


_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        _TOK.model_max_length = 1 << 40
    return _TOK


def _tokenize_book(args):
    path, row, ids_dir, key = args
    import pyarrow.parquet as pq
    text = pq.read_table(path, columns=["text"]).slice(row, 1)["text"][0].as_py()
    ids = np.asarray(_tokenizer()(text, add_special_tokens=False).input_ids, dtype=np.uint32)
    np.save(os.path.join(ids_dir, f"{key}.npy"), ids)
    words = text.split()
    return (key, int(ids.shape[0]), hashlib.sha256(text.encode("utf-8")).hexdigest(),
            {n: anchored_hashes(words, n) for n in BYTE_NS})


def training_datasets(config_paths: list[str]) -> list[dict]:
    """Every dataset directory a training config reads (sources + retention), with the config it came from."""
    out, seen = [], set()
    for cp in config_paths:
        c = json.load(open(cp))
        dirs = [(s["dataset_dir"], s.get("split", "train"), s.get("name")) for s in c.get("sources", [])]
        if c.get("retention"):
            r = c["retention"]
            dirs.append((r["dir"], r.get("split", "train"), "retention"))
        for d, split, name in dirs:
            key = (os.path.realpath(d), split)
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(dataset_dir=d, split=split, name=name, config=cp))
    return out


_SCAN = {}


def _scan_init(cand_hashes, cand_owner, cand_pos, W, byte_sets):
    _SCAN.update(h=cand_hashes, owner=cand_owner, pos=cand_pos, W=W, byte_sets=byte_sets)


def _scan_rows(args):
    """One batch of index items of one dataset: identity ids + token n-gram hits + anchored byte n-gram hits."""
    ds, items = args
    h, owner, pos, W = _SCAN["h"], _SCAN["owner"], _SCAN["pos"], _SCAN["W"]
    bsets = _SCAN["byte_sets"]
    ids_seen, urls, shas = set(), set(), set()
    hits, n_tok, n_rows, n_noids = [], 0, 0, 0
    byte_hits = []                                  # (sample_id, path, n, [matched hashes])
    n_words = 0
    for it in items:
        with open(os.path.join(ds, it["path"]), "rb") as f:
            f.seek(int(it["byte_offset"]))
            b = f.read(int(it["byte_length"]))
        n_rows += 1
        for m in PG19_ID_RE.finditer(b):
            ids_seen.add((m.group(1).decode(), int(m.group(2))))
        for m in URL_RE.finditer(b):
            urls.add(m.group(1).decode())
        for m in SHA_RE.finditer(b):
            shas.add(m.group(1).decode())
        m = IDS_RE.search(b)
        if m is None:
            n_noids += 1
            continue
        e = b.index(b"]", m.end())
        ids = np.array(json.loads(b[m.end() - 1:e + 1]), dtype=np.int64)
        n_tok += int(ids.shape[0])
        words = _tokenizer().decode(ids.tolist(), skip_special_tokens=False).split()
        n_words += len(words)
        for n in BYTE_NS:
            bh = anchored_hashes(words, n)
            if bh.size:
                arr = bsets[n]
                ix = np.minimum(np.searchsorted(arr, bh), arr.size - 1)
                mh = np.unique(bh[arr[ix] == bh])
                if mh.size:
                    byte_hits.append((it.get("sample_id"), it["path"], n, [int(x) for x in mh]))
        rh = ngram_hashes(ids, W)
        if rh.shape[0] == 0:
            continue
        k = np.searchsorted(h, rh)
        k = np.minimum(k, h.shape[0] - 1)
        hit = np.nonzero(h[k] == rh)[0]
        if hit.size:
            for j in np.unique(k[hit]):
                hits.append(dict(sample_id=it.get("sample_id"), path=it["path"], book=owner[j],
                                 book_pos=int(pos[j])))
    return dict(dataset=ds, ids=sorted(ids_seen), urls=sorted(urls), shas=sorted(shas), hits=hits, n_tok=n_tok,
                n_rows=n_rows, n_noids=n_noids, byte_hits=byte_hits, n_words=n_words)


def byte_index(byte_side: dict[str, dict]) -> dict:
    """byte_side: book key -> {n: anchored hashes}."""
    keys = sorted(byte_side)
    out = {}
    for n in BYTE_NS:
        hs = [byte_side[k][n] for k in keys]
        own = np.concatenate([np.full(x.shape[0], i, dtype=np.int64) for i, x in enumerate(hs)]) if hs else \
            np.zeros(0, np.int64)
        h = np.concatenate(hs) if hs else np.zeros(0, np.uint64)
        o = np.argsort(h, kind="stable")
        out[n] = dict(uniq=np.unique(h), h=h[o], owner=own[o], keys=keys,
                      per_book={k: int(np.unique(x).shape[0]) for k, x in zip(keys, hs)})
    return out


def byte_hits_by_book(bidx: dict, n: int, hashes) -> dict[str, list[int]]:
    """Matched row hashes -> {book key: the distinct matched n-gram hashes that book contains}."""
    d = bidx[n]
    hs = np.unique(np.asarray(hashes, dtype=np.uint64))
    lo = np.searchsorted(d["h"], hs, side="left")
    hi = np.searchsorted(d["h"], hs, side="right")
    out = {}
    for x, a, b in zip(hs, lo, hi):
        for k in {d["keys"][int(d["owner"][j])] for j in range(a, b)}:
            out.setdefault(k, []).append(int(x))
    return out


def scan_training(datasets: list[dict], cand: dict[str, np.ndarray], W, workers: int, batch: int = 64,
                  bidx: dict | None = None) -> dict:
    """cand: book key -> token ids; bidx: byte_index() of the same books."""
    owners, poss, hs = [], [], []
    per_book = {}
    for key, ids in cand.items():
        p, hh = sampled_ngrams(ids, W)
        per_book[key] = int(p.shape[0])
        owners += [key] * p.shape[0]
        poss.append(p)
        hs.append(hh)
    h = np.concatenate(hs)
    pos = np.concatenate(poss)
    order = np.argsort(h, kind="stable")
    h, pos = h[order], pos[order]
    owner = np.asarray(owners, dtype=object)[order]
    jobs = []
    for d in datasets:
        idx = Path(d["dataset_dir"]) / f"{d['split']}.records.idx.jsonl"
        items = [json.loads(l) for l in open(idx) if l.strip()]
        d["n_index_rows"] = len(items)
        d["index_sha256"] = sha256_file(idx)
        for a in range(0, len(items), batch):
            jobs.append((d["dataset_dir"], items[a:a + batch]))
    t0 = time.time()
    res = {d["dataset_dir"]: dict(ids=set(), urls=set(), shas=set(), hits=[], n_tok=0, n_rows=0, n_noids=0,
                                  byte_hits=[], n_words=0)
           for d in datasets}
    byte_sets = {n: bidx[n]["uniq"] for n in BYTE_NS}
    with Pool(workers, initializer=_scan_init, initargs=(h, owner, pos, W, byte_sets)) as p:
        for i, r in enumerate(p.imap_unordered(_scan_rows, jobs, chunksize=1)):
            a = res[r["dataset"]]
            a["ids"].update(map(tuple, r["ids"]))
            a["urls"].update(r["urls"])
            a["shas"].update(r["shas"])
            a["hits"] += r["hits"]
            for sid, path, n, mh in r["byte_hits"]:
                for bk, hb in sorted(byte_hits_by_book(bidx, n, mh).items()):
                    a["byte_hits"].append(dict(sample_id=sid, path=path, n=n, book=bk, n_ngrams=len(hb), ngrams=hb))
            for k in ("n_tok", "n_rows", "n_noids", "n_words"):
                a[k] += r[k]
            if (i + 1) % 100 == 0:
                print(f"[scan] {i + 1}/{len(jobs)} batches {time.time() - t0:.0f}s", flush=True)
    return dict(per_dataset=res, sampled_ngrams_per_book=per_book, wall_s=round(time.time() - t0, 1),
                byte_anchored_per_book={n: bidx[n]["per_book"] for n in BYTE_NS})


def ledger_records(ledger_path: str) -> dict:
    out = {}
    if not Path(ledger_path).exists():
        return out
    for l in open(ledger_path):
        r = json.loads(l)
        m = PG19_ID_RE.search(r["source_document_id"].encode())
        if m:
            out[(m.group(1).decode(), int(m.group(2)))] = dict(status=r.get("status"), classes=r.get("classes"))
    return out


def select(books: list[dict], n_main: int = N_MAIN) -> dict:
    """books: candidates (n_tokens, title, url, eligible)."""
    elig = [b for b in books if b["eligible"]]
    by_group = {}
    for b in sorted(elig, key=lambda b: rank_key(b["url"])):
        by_group.setdefault(b["group"], []).append(b["book_id"])
    url = {b["book_id"]: b["url"] for b in elig}
    reps = sorted((rank_key(url[m[0]]), g, m[0]) for g, m in by_group.items())
    main = [bid for _, _, bid in reps[:n_main]]
    two_m = sorted((b["book_id"] for b in elig if b["n_tokens"] >= E_2M), key=lambda bid: rank_key(url[bid]))
    return dict(main=main, two_m=two_m, groups={g: m for g, m in by_group.items()},
                group_rank=[dict(rank=i, group=g, representative=bid) for i, (_, g, bid) in enumerate(reps)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parquet-dir", default=PG19_PARQUET)
    ap.add_argument("--test-parquet", default=None, help="PG-19 test split parquet (secondary set); skipped if absent")
    ap.add_argument("--train-config", nargs="*", default=[D_CONFIG],
                    help="config_used.json of every run whose training data must not contain a chosen book")
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--work", default=None, help="work dir for the full-book token arrays (default <out>/work)")
    args = ap.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "recsa":
        sys.exit("run inside the recsa conda env (conda activate recsa)")
    out = Path(args.out)
    work = Path(args.work) if args.work else out / "work"
    (work / "ids").mkdir(parents=True, exist_ok=True)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pool = index_pool(args.parquet_dir)
    pre = [r for r in pool if r["n_chars"] >= CHAR_PREFILTER]
    print(f"[pool] {len(pool)} train books, {len(pre)} with >= {CHAR_PREFILTER:,} chars", flush=True)
    jobs = [(os.path.join(args.parquet_dir, r["file"]), r["row"], str(work / "ids"), r["book_id"]) for r in pre]
    test_rows = []
    if args.test_parquet and Path(args.test_parquet).exists():
        import pyarrow.parquet as pq
        tt = pq.read_table(args.test_parquet, columns=["short_book_title", "publication_date", "url"])
        for i, (ti, pd, u) in enumerate(zip(tt["short_book_title"].to_pylist(), tt["publication_date"].to_pylist(),
                                            tt["url"].to_pylist())):
            key = f"test_{gutenberg_no(u) or i}"
            test_rows.append(dict(row=i, book_id=key, title=ti, pub=pd, url=u))
            jobs.append((args.test_parquet, i, str(work / "ids"), key))
    with Pool(min(args.workers, 32)) as p:
        res_tok = p.map(_tokenize_book, jobs, chunksize=1)
    tok = {k: (n, sh) for k, n, sh, _ in res_tok}
    byte_side_all = {k: bs for k, _, _, bs in res_tok}
    for r in pre:
        r["n_tokens"], r["text_sha256"] = tok[r["book_id"]]
    for r in test_rows:
        r["n_tokens"], r["text_sha256"] = tok[r["book_id"]]
    print(f"[tok] {len(jobs)} books tokenized {time.time() - t0:.0f}s", flush=True)
    cands = [r for r in pre if r["n_tokens"] >= E_MAIN]

    datasets = training_datasets(args.train_config)
    W = ngram_weights()
    cand_ids = {r["book_id"]: np.load(work / "ids" / f"{r['book_id']}.npy").astype(np.int64)
                for r in cands + test_rows}
    bidx = byte_index({k: byte_side_all[k] for k in cand_ids})
    del byte_side_all
    scan = scan_training(datasets, cand_ids, W, args.workers, bidx=bidx)
    ids_all, urls_all, shas_all = set(), set(), set()
    hits_by_book = {}
    byte_by_book = {}
    for ds, a in scan["per_dataset"].items():
        ids_all |= a["ids"]
        urls_all |= a["urls"]
        shas_all |= a["shas"]
        for h in a["hits"]:
            hits_by_book.setdefault(h["book"], []).append(dict(dataset=ds, **h))
        for h in a["byte_hits"]:
            e = byte_by_book.setdefault(h["book"], {}).setdefault(h["n"], dict(rows=set(), ngrams=set(), examples=[]))
            e["rows"].add((ds, h["sample_id"]))
            e["ngrams"].update(h["ngrams"])
            if len(e["examples"]) < 5:
                e["examples"].append(dict(dataset=ds, sample_id=h["sample_id"], path=h["path"], n_ngrams=h["n_ngrams"]))
    pool_by_id = {(r["file"], r["row"]): r for r in pool}
    train_titles = {norm_title(pool_by_id[i]["title"]) for i in ids_all if i in pool_by_id}
    train_urls = {gutenberg_no(pool_by_id[i]["url"]) for i in ids_all if i in pool_by_id} | urls_all
    ledger = ledger_records(args.ledger)

    def decontam(r, test=False):
        fid = None if test else (r["file"], r["row"])
        led = ledger.get(fid) if fid else None
        hb = hits_by_book.get(r["book_id"], [])
        bb = byte_by_book.get(r["book_id"], {})
        b50, b13 = bb.get(BYTE_RULE_N, {}), bb.get(13, {})
        rec = dict(
            id_in_training=bool(fid in ids_all) if fid else False,
            url_in_training=gutenberg_no(r["url"]) in train_urls,
            title_in_training=norm_title(r["title"]) in train_titles,
            sha_in_training=r["text_sha256"] in shas_all,
            ngram_hits=len(hb), ngram_hit_rows=len({(h["dataset"], h["sample_id"]) for h in hb}),
            ngram_hit_examples=hb[:5], sampled_ngrams=scan["sampled_ngrams_per_book"].get(r["book_id"]),
            byte50_ngram_hits=len(b50.get("ngrams", ())), byte50_hit_rows=len(b50.get("rows", ())),
            byte50_hit_examples=b50.get("examples", []),
            byte50_anchored=scan["byte_anchored_per_book"][BYTE_RULE_N].get(r["book_id"]),
            byte13_rows_report_only=len(b13.get("rows", ())), byte13_ngrams_report_only=len(b13.get("ngrams", ())),
            ledger_status=(led or {}).get("status"), ledger_classes=(led or {}).get("classes"),
            pooler_coopt_content=bool(led and "POOLER_COOPT_CONTENT" in (led.get("classes") or [])))
        rec["clean_token_rule_only"] = not (rec["id_in_training"] or rec["url_in_training"] or rec["title_in_training"]
                                            or rec["sha_in_training"] or rec["ngram_hits"] or rec["pooler_coopt_content"])
        rec["clean"] = rec["clean_token_rule_only"] and rec["byte50_ngram_hits"] == 0
        return rec

    for r in cands:
        r["decontam"] = decontam(r)
        r["reference_work"] = bool(REFERENCE_RE.search(r["title"] or ""))
        r["group"] = group_of(r["title"], r["book_id"])
        r["rank_key"] = rank_key(r["url"])
        r["eligible"] = r["decontam"]["clean"] and not r["reference_work"]
        r["widely_quoted"] = r["decontam"]["byte13_rows_report_only"] >= WIDELY_QUOTED_ROWS13
    for r in test_rows:
        r["decontam"] = decontam(r, test=True)
    sel = select(cands)
    if len(sel["main"]) < N_MAIN:
        print(f"[select] only {len(sel['main'])} eligible work groups (< N_MAIN={N_MAIN}): the MAIN set has "
              f"{len(sel['main'])} books; no rule is relaxed", flush=True)

    # token files + manifest
    manifest = {}

    def save(name, arr):
        p = out / "tokens" / name
        np.save(p, np.ascontiguousarray(arr, dtype=np.uint32))
        manifest[f"tokens/{name}"] = sha256_file(p)

    for bid in sel["main"]:
        save(f"{bid}_E{E_MAIN}.npy", np.load(work / "ids" / f"{bid}.npy")[:E_MAIN])
    for bid in sel["two_m"]:
        save(f"{bid}_E{E_2M}.npy", np.load(work / "ids" / f"{bid}.npy")[:E_2M])
    test_set = []
    for r in test_rows:
        if r["n_tokens"] >= TEST_MIN and r["decontam"]["clean"]:
            save(f"{r['book_id']}_first.npy", np.load(work / "ids" / f"{r['book_id']}.npy")[:TEST_FIRST])
            test_set.append(r["book_id"])
    test_lengths = [8192, 16384, 32768, 65536, 131072]
    docset = dict(
        schema="ppl_docset/v2", built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        schema_note="byte-level anchored 50-word rule (token OR byte hit "
                    "excludes, main and test); MAIN = all eligible groups when fewer than n_main",
        rules=dict(seed=SEED, n_main=N_MAIN, E_main=E_MAIN, E_2m=E_2M, char_prefilter=CHAR_PREFILTER,
                   reference_re=REFERENCE_RE.pattern, groups={n: rx.pattern for n, rx in GROUPS},
                   ngram=NGRAM, stride=STRIDE, min_distinct=MIN_DISTINCT, test_first=TEST_FIRST, test_min=TEST_MIN,
                   byte_rule=dict(n_words=BYTE_RULE_N, anchor=f"crc32(first word) % {ANCHOR_MOD} == 0",
                                  hash="blake2b digest_size 8 of ' '.join(words)",
                                  words="str.split() of the raw book text / of tokenizer.decode(row input_ids)"),
                   report_only=dict(byte_n_words=13, widely_quoted_rows13=WIDELY_QUOTED_ROWS13),
                   exclusion="identity OR token 50-gram OR byte 50-word OR a source-ledger exposure",
                   order="sha256(seed:url) ascending; group representative = first-ranked member"),
        n_main=len(sel["main"]), n_main_requested=N_MAIN,
        main_shortfall=(None if len(sel["main"]) >= N_MAIN else
                        f"only {len(sel['main'])} eligible work groups; the MAIN set is all of them"),
        tokenizer=dict(path=TOKENIZER, tokenizer_json_md5=hashlib.md5(
            open(Path(TOKENIZER) / "tokenizer.json", "rb").read()).hexdigest(), add_special_tokens=False),
        parquet_dir=args.parquet_dir, test_parquet=args.test_parquet,
        test_parquet_sha256=sha256_file(args.test_parquet) if args.test_parquet and Path(
            args.test_parquet).exists() else None,
        train_configs=args.train_config, ledger=args.ledger,
        main=[next(dict(r, decontam=None) for r in cands if r["book_id"] == b) for b in sel["main"]],
        two_m=[next(dict(r, decontam=None) for r in cands if r["book_id"] == b) for b in sel["two_m"]],
        test=dict(books=test_set, lengths=test_lengths,
                  n_qualifying={str(L): sum(1 for r in test_rows if r["book_id"] in test_set and r["n_tokens"] >= L)
                                for L in test_lengths},
                  n_test_books=len(test_rows),
                  n_excluded_decontam=sum(1 for r in test_rows if not r["decontam"]["clean"]),
                  n_too_short=sum(1 for r in test_rows if r["n_tokens"] < TEST_MIN),
                  rows=test_rows),
        group_rank=sel["group_rank"], candidates=cands, n_pool=len(pool), n_prefilter=len(pre),
        n_candidates_1m=len(cands), n_eligible=sum(r["eligible"] for r in cands))
    (out / "docset.json").write_text(json.dumps(docset, indent=1))
    dec = dict(datasets=datasets, wall_s=scan["wall_s"],
               per_dataset={ds: dict(n_rows=a["n_rows"], n_tokens=a["n_tok"], rows_without_input_ids=a["n_noids"],
                                     pg19_ids=len(a["ids"]), gutenberg_urls=len(a["urls"]), source_sha256=len(a["shas"]),
                                     ngram_hits=len(a["hits"]), n_words_decoded=a["n_words"],
                                     byte50_hits=sum(1 for h in a["byte_hits"] if h["n"] == BYTE_RULE_N),
                                     byte13_hits=sum(1 for h in a["byte_hits"] if h["n"] == 13))
                            for ds, a in scan["per_dataset"].items()},
               byte_hits_n50=[dict(dataset=ds, **{k: v for k, v in h.items() if k != "ngrams"})
                              for ds, a in scan["per_dataset"].items() for h in a["byte_hits"] if h["n"] == BYTE_RULE_N],
               training_pg19_ids=sorted(f"{f}:{r}" for f, r in ids_all),
               hits=[dict(dataset=ds, **h) for ds, a in scan["per_dataset"].items() for h in a["hits"]])
    (out / "decontam.json").write_text(json.dumps(dec, indent=1))
    manifest["docset.json"] = sha256_file(out / "docset.json")
    (out / "MANIFEST.sha256").write_text("".join(f"{v}  {k}\n" for k, v in sorted(manifest.items())))
    print(json.dumps(dict(main=sel["main"], two_m=sel["two_m"], test=len(test_set), n_eligible=docset["n_eligible"],
                          test_excluded=[r["book_id"] for r in test_rows if not r["decontam"]["clean"]],
                          n_qualifying=docset["test"]["n_qualifying"], wall_s=round(time.time() - t0, 1))), flush=True)
    print("PPL-DOCSET-DONE", flush=True)


if __name__ == "__main__":
    main()
