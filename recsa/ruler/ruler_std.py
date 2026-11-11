"""Standard-format RULER suite generator (13 tasks) at true token lengths; eval/ruler/gen_suites.sh runs it."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import uuid
from pathlib import Path

POSITION_BUCKETS = ["early", "middle", "late"]
BUCKET_RANGES = {"early": (0.05, 0.33), "middle": (0.34, 0.66), "late": (0.67, 0.95)}

NOISE_LINE = ("The grass is green. The sky is blue. The sun is yellow. "
              "Here we go. There and back again.")

ADJS = """curious quiet brave calm eager fancy gentle happy jolly kind lively
merry nice proud silly witty zealous bright clever daring elegant fierce
graceful humble inventive keen loyal modest noble patient quick rustic sturdy
tender upbeat vivid warm youthful ancient bold crisp deep earnest firm glad
hardy ideal just mellow neat open polite rare solid true usual vast wise""".split()

NOUNS = """thimble lantern walnut compass ribbon anchor bucket candle drawer
easel feather goblet hammer inkwell jacket kettle ladder mirror needle oven
pencil quilt rocket saddle teapot umbrella violin wagon xylophone yarn zipper
basket cabinet dolphin engine falcon garden harbor island jungle kitchen lemon
meadow notebook orchard palace quarry river shovel tunnel valley window""".split()

COMMON_WORDS = """time year people way day man thing woman life child world
school state family student group country problem hand part place case week
company system program question work government number night point home water
room mother area money story fact month lot right study book eye job word
business issue side kind head house service friend father power hour game line
end member law car city community name president team minute idea body
information back parent face others level office door health person art war
history party result change morning reason research girl guy moment air
teacher force education""".split()

SYLLABLES = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]

TASKS = {
    "niah_single_1":   dict(kind="niah", haystack="noise", vtype="number", nk=1, nv=1, nq=1, mnt=128),
    "niah_single_2":   dict(kind="niah", haystack="essay", vtype="number", nk=1, nv=1, nq=1, mnt=128),
    "niah_single_3":   dict(kind="niah", haystack="essay", vtype="uuid",   nk=1, nv=1, nq=1, mnt=128),
    "niah_multikey_1": dict(kind="niah", haystack="essay", vtype="number", nk=4, nv=1, nq=1, mnt=128),
    "niah_multikey_2": dict(kind="niah", haystack="kvlines", ktype="word", vtype="number", nk=1, nv=1, nq=1, mnt=128),
    "niah_multikey_3": dict(kind="niah", haystack="kvlines", ktype="uuid", vtype="uuid",   nk=1, nv=1, nq=1, mnt=128),
    "niah_multivalue": dict(kind="niah", haystack="essay", vtype="number", nk=1, nv=4, nq=1, mnt=128),
    "niah_multiquery": dict(kind="niah", haystack="essay", vtype="number", nk=4, nv=1, nq=4, mnt=128),
    "vt":  dict(kind="vt", chains=1, hops=4, mnt=30),
    "cwe": dict(kind="cwe", n_common=10, freq_cw=30, freq_ucw=3, mnt=120),
    "fwe": dict(kind="fwe", alpha=2.0, mnt=50),
    "qa_1": dict(kind="qa", source="squad", mnt=32),
    "qa_2": dict(kind="qa", source="hotpot", mnt=32),
}

NIAH_PREAMBLE = ("Some special magic {vp} are hidden within the following text. "
                 "Make sure to memorize it. I will quiz you about the {vp} afterwards.\n")
NIAH_QUESTION = ("\nWhat are all the special magic {vp} for {query} mentioned in the "
                 "provided text? The special magic {vp} for {query} mentioned in the "
                 "provided text are")
NIAH_NEEDLE = "One of the special magic {vp} for {key} is: {value}."

VT_TEXT = ("Memorize and track the chain(s) of variable assignment hidden in the "
           "following text.\n\n{context}\nQuestion: Find all variables that are "
           "assigned the value {value} in the text above.\nAnswer: According to the "
           "chain(s) of variable assignment in the text above, {n} variables are "
           "assigned the value {value}, they are:")

CWE_TEXT = ("Below is a numbered list of words. In these words, some appear more "
            "often than others. Memorize the ones that appear most often.\n{context}"
            "\nQuestion: What are the 10 most common words in the above list?\n"
            "Answer: The top 10 words that appear most often in the list are:")

FWE_TEXT = ("Read the following coined words. Memorize the top 3 most frequent "
            "coined words. {context}\nQuestion: Do not provide any explanation. "
            "Please ignore the dots '....'. What are the 3 most frequently appeared "
            "words in the above coined words?\nAnswer: According to the coined words "
            "in the above, the 3 most frequently appeared words are:")

QA_TEXT = ("Answer the question based on the given documents. Only give me the "
           "answer and do not output any other words.\n\nThe following are given "
           "documents.\n\n{context}\n\nAnswer the question based on the given "
           "documents. Only give me the answer and do not output any other words."
           "\n\nQuestion: {question} Answer:")

# ---------------------------------------------------------------- primitives

def _vp(vtype):
    return {"number": "numbers", "uuid": "uuids"}[vtype]


def _value(rng, vtype):
    if vtype == "number":
        return str(rng.randint(1000000, 9999999))
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _word_key(rng, taken, suffix=None):
    while True:
        k = f"{rng.choice(ADJS)}-{rng.choice(NOUNS)}"
        if suffix is not None:
            k = f"{k}-{suffix}"
        if k not in taken:
            taken.add(k)
            return k


def _coined(rng, taken):
    while True:
        w = "".join(rng.choice(SYLLABLES) for _ in range(3))
        if w not in taken:
            taken.add(w)
            return w


def _depth(rng, bucket):
    lo, hi = BUCKET_RANGES[bucket]
    return rng.uniform(lo, hi)


def _insert(units, needles_at):
    """needles_at: list of (depth_fraction, line)."""
    out = list(units)
    for d, line in sorted(needles_at, key=lambda x: -x[0]):
        out.insert(min(len(out), max(0, round(d * len(out)))), line)
    return "\n".join(out)


def _essay_units(pool, rng, n_chars):
    start = rng.randrange(len(pool))
    units, tot = [], 0
    i = start
    while tot < n_chars:
        units.append(pool[i % len(pool)])
        tot += len(pool[i % len(pool)]) + 1
        i += 1
    return units


def _build_niah(task, cfg, rng, bucket, pool):
    vp = _vp(cfg["vtype"])
    taken = set()
    if cfg["haystack"] == "kvlines":
        ktype = cfg.get("ktype", "word")
        if ktype == "uuid":
            gold_key = _value(rng, "uuid")
        else:
            gold_key = _word_key(rng, taken)
        gold_vals = [_value(rng, cfg["vtype"])]
        d = _depth(rng, bucket)
        gold_line = NIAH_NEEDLE.format(vp=vp, key=gold_key, value=gold_vals[0])
        seed_lines = rng.random()

        def make(u):
            r = random.Random(seed_lines)
            tk = set(taken)
            lines = []
            for j in range(u):
                k = (_value(r, "uuid") if ktype == "uuid"
                     else _word_key(r, tk, suffix=j))
                lines.append(NIAH_NEEDLE.format(vp=vp, key=k, value=_value(r, cfg["vtype"])))
            ctx = _insert(lines, [(d, gold_line)])
            text = (NIAH_PREAMBLE.format(vp=vp) + ctx
                    + NIAH_QUESTION.format(vp=vp, query=gold_key))
            return text, gold_vals, dict(needle_depth=round(d, 4), query=gold_key)
        return make, 60000

    # essay / noise haystacks
    keys = [_word_key(rng, taken) for _ in range(cfg["nk"])]
    queried = keys[:cfg["nq"]]
    needles, golds = [], []
    if cfg["nv"] > 1:                       # multivalue: one key, nv values
        for _ in range(cfg["nv"]):
            v = _value(rng, cfg["vtype"])
            golds.append(v)
            needles.append((rng.uniform(0.05, 0.95),
                            NIAH_NEEDLE.format(vp=vp, key=keys[0], value=v)))
    elif cfg["nk"] > 1:                     # multikey / multiquery
        d0 = _depth(rng, bucket) if cfg["nq"] == 1 else None
        for j, k in enumerate(keys):
            v = _value(rng, cfg["vtype"])
            if j < cfg["nq"]:
                golds.append(v)
            d = d0 if (j == 0 and d0 is not None) else rng.uniform(0.05, 0.95)
            needles.append((d, NIAH_NEEDLE.format(vp=vp, key=k, value=v)))
    else:                                   # single needle
        v = _value(rng, cfg["vtype"])
        golds.append(v)
        needles.append((_depth(rng, bucket), NIAH_NEEDLE.format(vp=vp, key=keys[0], value=v)))

    if cfg["nq"] == 1:
        query = queried[0]
    else:
        query = ", ".join(queried[:-1]) + " and " + queried[-1]
    seed_fill = rng.random()

    def make(u):
        if cfg["haystack"] == "noise":
            units = [NOISE_LINE] * u
        else:
            units = _essay_units(pool, random.Random(seed_fill), u * 100)
        ctx = _insert(units, needles)
        text = (NIAH_PREAMBLE.format(vp=vp) + ctx
                + NIAH_QUESTION.format(vp=vp, query=query))
        meta = dict(needle_depths=[round(d, 4) for d, _ in needles], query=query)
        return text, golds, meta
    return make, 30000


def _build_vt(task, cfg, rng, bucket, pool):
    n_vars = cfg["hops"] + 1
    taken = set()
    names = []
    while len(names) < n_vars:
        n = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
        if n not in taken:
            taken.add(n)
            names.append(n)
    value = str(rng.randint(10000, 99999))
    stmts = [f"VAR {names[0]} = {value}"] + [
        f"VAR {names[j]} = VAR {names[j - 1]}" for j in range(1, n_vars)]
    # unsorted: statements land in random textual order, as in upstream vt
    depths = [rng.uniform(0.05, 0.95) for _ in stmts]

    def make(u):
        ctx = _insert([NOISE_LINE] * u, list(zip(depths, stmts)))
        text = VT_TEXT.format(context=ctx, value=value, n=n_vars)
        return text, list(names), dict(value=value)
    return make, 30000


def _build_cwe(task, cfg, rng, bucket, pool):
    common = rng.sample(COMMON_WORDS, cfg["n_common"])
    seed_fill = rng.random()

    def make(u):
        r = random.Random(seed_fill)
        tk = set(common)
        items = common * cfg["freq_cw"]
        for _ in range(u):
            items.extend([_coined(r, tk)] * cfg["freq_ucw"])
        r.shuffle(items)
        ctx = "\n".join(f"{j + 1}. {w}" for j, w in enumerate(items))
        return CWE_TEXT.format(context=ctx), list(common), dict(n_items=len(items))
    return make, 20000


def _build_fwe(task, cfg, rng, bucket, pool):
    seed_fill = rng.random()

    def make(u):
        r = random.Random(seed_fill)
        vocab_n = max(12, u // 64)
        tk = set()
        vocab = [_coined(r, tk) for _ in range(vocab_n)]
        h = sum(1.0 / (j + 1) ** cfg["alpha"] for j in range(vocab_n))
        counts = [max(1, round(u / h / (j + 1) ** cfg["alpha"])) for j in range(vocab_n)]
        counts[2] = max(counts[2], counts[3] + 2) if vocab_n > 3 else counts[2]
        items = [w for w, c in zip(vocab, counts) for _ in range(c)]
        r.shuffle(items)
        ctx = " .... ".join(items)
        return FWE_TEXT.format(context=ctx), vocab[:3], dict(n_items=len(items), vocab=vocab_n)
    return make, 60000


def _build_qa(task, cfg, rng, bucket, qa_pool):
    golds_docs, question, answers = qa_pool.pick(rng)
    d = _depth(rng, bucket)
    seed_fill = rng.random()

    def make(u):
        r = random.Random(seed_fill)
        docs = qa_pool.distractors(r, u, exclude=golds_docs)
        for j, g in enumerate(golds_docs):
            docs.insert(min(len(docs), round(d * len(docs)) + j), g)
        ctx = "\n\n".join(f"Document {j + 1}:\n{t}" for j, t in enumerate(docs))
        return (QA_TEXT.format(context=ctx, question=question), answers,
                dict(gold_depth=round(d, 4), n_gold_docs=len(golds_docs)))
    return make, 3000


class _QAPool:
    """Flattened (gold_docs, question, answers) triples + distractor doc pool."""

    def __init__(self, source, path):
        self.items, self.docs = [], []
        data = json.loads(Path(path).read_text())
        if source == "squad":
            for art in data["data"]:
                for para in art["paragraphs"]:
                    self.docs.append(para["context"])
                    for qa in para["qas"]:
                        if qa.get("answers"):
                            answers = sorted({a["text"] for a in qa["answers"]})
                            self.items.append(([para["context"]], qa["question"], answers))
                            break
        else:  # hotpot dev distractor
            for row in data:
                titles = {t for t, _ in row["supporting_facts"]}
                golds = ["".join(sents) for t, sents in row["context"] if t in titles]
                self.docs.extend("".join(sents) for _, sents in row["context"])
                if golds and row.get("answer"):
                    self.items.append((golds, row["question"], [row["answer"]]))

    def pick(self, rng):
        return self.items[rng.randrange(len(self.items))]

    def distractors(self, rng, n, exclude):
        ex = set(exclude)
        cap = sum(1 for t in self.docs if t not in ex)
        out, seen = [], set()
        while len(out) < n:
            t = self.docs[rng.randrange(len(self.docs))]
            if t in ex or (id(t) in seen and len(seen) < cap):
                continue
            seen.add(id(t))
            out.append(t)
        return out

# ---------------------------------------------------------------- fitting

_G = {}


def _tokens(text):
    return len(_G["tok"](text, add_special_tokens=False)["input_ids"])


def _fit(make, budget):
    """Binary-search filler units so tokens(text) <= budget, maximal."""
    lo, hi = 1, 256
    while _tokens(make(hi)[0]) < budget:
        lo, hi = hi, hi * 2
        if hi > 4_000_000:
            raise RuntimeError("filler bracket blew up")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _tokens(make(mid)[0]) <= budget:
            lo = mid
        else:
            hi = mid
    return make(lo)

# ---------------------------------------------------------------- workers

QA_FILES = {"squad": "squad_dev_v1.1.json", "hotpot": "hotpot_dev_distractor_v1.json"}


def _init_worker(tokenizer_dir, filler_path, pool_chars, qa_dir, qa_sources=()):
    from transformers import AutoTokenizer
    _G["tok"] = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    pool, tot = [], 0
    with open(filler_path) as f:
        for line in f:
            t = json.loads(line)["text"].strip()
            if len(t) <= 200:
                continue
            words, piece = t.split(), []
            n = 0
            for w in words:
                piece.append(w)
                n += len(w) + 1
                if n >= 300:
                    pool.append(" ".join(piece))
                    piece, n = [], 0
            if piece:
                pool.append(" ".join(piece))
            tot += len(t)
            if tot >= pool_chars:
                break
    _G["pool"] = pool
    _G["qa"] = {s: _QAPool(s, Path(qa_dir) / QA_FILES[s]) for s in qa_sources}


def _gen_one(job):
    task, idx, L, seed, split, fraction = job
    cfg = TASKS[task]
    rng = random.Random(f"{seed}:{split}:{L}:{task}:{idx}")
    bucket = POSITION_BUCKETS[idx % 3]
    kind = cfg["kind"]
    if kind == "niah":
        make, _ = _build_niah(task, cfg, rng, bucket, _G["pool"])
    elif kind == "vt":
        make, _ = _build_vt(task, cfg, rng, bucket, _G["pool"])
    elif kind == "cwe":
        make, _ = _build_cwe(task, cfg, rng, bucket, _G["pool"])
    elif kind == "fwe":
        make, _ = _build_fwe(task, cfg, rng, bucket, _G["pool"])
    else:
        make, _ = _build_qa(task, cfg, rng, bucket, _G["qa"][cfg["source"]])
    if kind == "niah" and (cfg.get("nv", 1) > 1 or cfg.get("nq", 1) > 1):
        bucket = "na"  # multi-needle placement is spread, not bucketed

    budget = int(L * fraction)
    text, golds, meta = _fit(make, budget)
    n_tok = _tokens(text)
    if not 0.97 * budget <= n_tok <= 1.005 * budget:
        raise RuntimeError(f"{task}[{idx}]: {n_tok} tokens vs budget {budget}")
    return dict(
        sample_id=f"ruler13_{L}-{split}-{L}-{idx:06d}-{task}",
        split=split, source_type="ruler_std", task_type=task,
        text=text, prompt="", answer=golds[0], answers=golds,
        target_seq_len=L, actual_tokens=n_tok, seed=seed,
        position_bucket=bucket, difficulty=f"{L // 1024}k",
        metadata=dict(ruler_task=task, match="part" if kind == "qa" else "all",
                      max_new_tokens=cfg["mnt"], n_golds=len(golds),
                      template="ruler_base_raw", **meta))

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--samples-per-task", type=int, default=56)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--target-tokenizer", required=True)
    ap.add_argument("--token-fraction", type=float, default=1.0)
    ap.add_argument("--filler-pool", default=os.path.join(os.environ.get("RECSA_DATA", "third_party/data"),
                                                         "ruler_filler", "source_docs.jsonl"))
    ap.add_argument("--pool-chars", type=int, default=250_000_000)
    ap.add_argument("--qa-sources", default=os.path.join(os.environ.get("RECSA_DATA", "third_party/data"),
                                                        "ruler_sources"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    unknown = set(args.tasks) - set(TASKS)
    if unknown:
        raise SystemExit(f"unknown tasks: {sorted(unknown)}")
    out = Path(args.output_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    eval_seed = args.seed + 1  # the eval split's seed

    for L in args.lengths:
        jobs = [(t, i, L, eval_seed, "eval", args.token_fraction)
                for t in args.tasks for i in range(args.samples_per_task)]
        need_qa = tuple(sorted({TASKS[t]["source"] for t in args.tasks
                                if TASKS[t]["kind"] == "qa"}))
        init = (args.target_tokenizer, args.filler_pool, args.pool_chars,
                args.qa_sources, need_qa)
        if args.workers > 1:
            with mp.get_context("spawn").Pool(args.workers, _init_worker, init) as p:
                rows = p.map(_gen_one, jobs, chunksize=1)
        else:
            _init_worker(*init)
            rows = [_gen_one(j) for j in jobs]
        path = out / "samples" / f"eval_{L}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[ruler_std] wrote {len(rows)} rows -> {path}")

    manifest = dict(
        generator="recsa.ruler.ruler_std", seed=args.seed, eval_seed=eval_seed,
        lengths=args.lengths, tasks=args.tasks,
        samples_per_task=args.samples_per_task,
        token_fraction=args.token_fraction,
        target_tokenizer=args.target_tokenizer,
        filler_pool=args.filler_pool, pool_chars=args.pool_chars,
        qa_sources=args.qa_sources,
        format="ruler_base_raw (no chat template; completion-style cues)",
        scoring="containment string match; match=all -> mean over golds, "
                "match=part -> any gold; see ruler_eval.py",
        deviations=["essay haystack = DCLM filler_pool_v3, not PG essays",
                    "cwe/fwe filler words from deterministic syllable pool, "
                    "not wonderwords; cwe targets are real common words",
                    "multikey_2/3 keys get numeric suffix for uniqueness"],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ruler_std] manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
