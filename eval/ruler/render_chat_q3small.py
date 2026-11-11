#!/usr/bin/env python
"""usage: conda run -n recsa python render_chat_q3small.py --raw raw/samples/eval_L.jsonl --tokenizer DIR --out DIR --L
L --per-shard K [--pad 64] [--expect-template-sha256 HEX] [--dry-run]

--dry-run streams the raw file, renders only
the first 2 rows per task, prints their tails plus the would-be shard layout of the full suite, and writes nothing."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

TASK_ORDER = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
              "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]
THINK_EMPTY = "<think>\n\n</think>\n\n"
HYBRID_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
_SID = re.compile(r'^\{"sample_id": "[^"]*?-([a-z_0-9]+)"')   # task_type is the sample_id's last '-' field


def split(task, text):
    if task.startswith("niah"):
        i = text.rfind("? The special magic ")
        assert i > 0, task
        body, prefix = text[:i + 1], text[i + 2:]
        assert prefix.endswith("mentioned in the provided text are"), prefix[-80:]
    elif task in ("vt", "cwe", "fwe"):
        i = text.rfind("\nAnswer: ")
        assert i > 0, task
        body, prefix = text[:i], text[i + 1:]
        assert prefix.endswith(":") and "\n" not in prefix, prefix
    else:
        assert text.endswith(" Answer:"), text[-80:]
        body, prefix = text[:-len(" Answer:")], "Answer:"
        assert "\nQuestion: " in body[-2000:], task
    return body, prefix


def read_rows(path, dry_run):
    """Return (rows, n_total)."""
    rows, seen, n_total = [], {}, 0
    with open(path) as f:
        for line in f:
            n_total += 1
            if dry_run:
                m = _SID.match(line)
                task = m.group(1) if m else json.loads(line)["task_type"]
                if seen.get(task, 0) >= 2:
                    continue
                seen[task] = seen.get(task, 0) + 1
            rows.append(json.loads(line))
    return rows, n_total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--per-shard", type=int, required=True)
    ap.add_argument("--pad", type=int, default=64)
    ap.add_argument("--expect-template-sha256", default=HYBRID_TEMPLATE_SHA256)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    tpl_sha = hashlib.sha256(str(tok.chat_template or "").encode()).hexdigest()
    assert tpl_sha == a.expect_template_sha256, (
        f"{a.tokenizer}: chat_template sha256 {tpl_sha} != expected {a.expect_template_sha256} "
        "(not the hybrid-thinking Qwen3 template; a plain-template trunk takes the `chat` render)")
    tok_md5 = hashlib.md5(open(Path(a.tokenizer) / "tokenizer.json", "rb").read()).hexdigest()
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    probe = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    assert probe.endswith("<|im_start|>assistant\n" + THINK_EMPTY), repr(probe[-60:])
    rows, n_total = read_rows(a.raw, a.dry_run)
    assert a.dry_run or len(rows) == n_total
    rendered, worst = [], 0
    for r in rows:
        assert r["target_seq_len"] == a.L
        task = r["task_type"]
        body, prefix = split(task, r["text"])
        text = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) + prefix
        assert text.endswith(THINK_EMPTY + prefix) and text.count("<think>") == 1, (r["sample_id"], text[-200:])
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert ids.count(im_start) == 2, (r["sample_id"], ids.count(im_start))
        mnt = int(r["metadata"]["max_new_tokens"])
        tree_end = len(ids) + mnt
        assert tree_end + a.pad <= a.L, (r["sample_id"], len(ids), mnt, a.L)
        worst = max(worst, tree_end)
        m = dict(r["metadata"])
        m.update(template="ruler_chat_q3small_v1 (user=body, assistant opens with the EMPTY think block "
                          "(enable_thinking=False) then answer_prefix)",
                 answer_prefix=prefix, max_new_tokens=mnt, answer_prefix_in_text=True, enable_thinking=False,
                 think_prefix_in_text=THINK_EMPTY, raw_actual_tokens=r["actual_tokens"],
                 raw_text_sha256=hashlib.sha256(r["text"].encode()).hexdigest())
        rr = dict(r)
        rr.update(text=text, actual_tokens=len(ids), metadata=m)
        rendered.append(rr)
    idx = lambda r: int(r["sample_id"].split("-")[3])  # noqa: E731
    rendered.sort(key=lambda r: (idx(r), TASK_ORDER.index(r["task_type"])))
    toks = [r["actual_tokens"] for r in rendered]
    n_sh = (n_total + a.per_shard - 1) // a.per_shard
    meta = dict(L=a.L, rows=n_total, shards=n_sh, per_shard=a.per_shard, tokens_min=min(toks),
                tokens_max=max(toks), worst_case_tree_end=worst, pad=a.pad, raw=a.raw, tokenizer=a.tokenizer,
                render="ruler_chat_q3small_v1", enable_thinking=False, think_prefix=THINK_EMPTY, mnt_extra=0,
                chat_template_sha256=tpl_sha, tokenizer_json_md5=tok_md5,
                tokenizer_note="hybrid-thinking Qwen3 template (sha256 asserted) over the tokenizer.json shared by "
                               "Qwen3-0.6B/1.7B/4B; one render serves every trunk with "
                               "this template hash",
                shard_order="index-major, tasks in TASK_ORDER within an index")
    if a.dry_run:
        for r in rendered[::2]:
            print(r["task_type"], r["metadata"]["raw_actual_tokens"], "->", r["actual_tokens"],
                  "| tail", repr(r["text"][-120:]))
        meta.update(dry_run=True, rows_rendered=len(rendered))
        print(json.dumps(meta))
        return
    out = Path(a.out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    with open(out / f"eval_{a.L}.jsonl", "w") as f:
        for r in rendered:
            f.write(json.dumps(r) + "\n")
    for k in range(n_sh):
        with open(out / "shards" / f"s{a.L}_{k:03d}.jsonl", "w") as f:
            for r in rendered[k * a.per_shard:(k + 1) * a.per_shard]:
                f.write(json.dumps(r) + "\n")
    (out / f"render_{a.L}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
