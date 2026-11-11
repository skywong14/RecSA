"""Dense-attention RULER evaluation with the suite, prompts and scoring of the RecSA evaluation
(scripts/ruler_eval.py)."""
import argparse
import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from recsa.ruler.rulerlite import normalize_prediction  # noqa: E402
import env_assert  # noqa: E402
from ruler_eval import (score_row, score_row_tok,  # noqa: E402
                                 score_row_tok_any,
                                 score_row_tok_credited, sha256)


def score_bnd(pred: str, golds, match: str) -> float:
    p = pred.lower()
    hits = []
    for g in golds:
        s = g.lower().strip()
        hits.append(1.0 if re.search(r"(?<![0-9a-z_])" + re.escape(s) + r"(?![0-9a-z_])", p)
                    else 0.0)
    return max(hits) if match == "part" else sum(hits) / len(hits)


def split_device_map(n_layers, split):
    """--device-map split: decoder layers [b_g, b_{g+1}) on GPU g, embed_tokens and the tied lm_head on GPU 0, final
    norm and rotary_emb on the last GPU."""
    if len(split) < 1 or any(n < 1 for n in split) or sum(split) != n_layers:
        raise SystemExit(f"--layer-split {split}: need positive counts summing to {n_layers} layers")
    dm = {"model.embed_tokens": 0, "lm_head": 0}
    layer = 0
    for g, n in enumerate(split):
        for _ in range(n):
            dm[f"model.layers.{layer}"] = g
            layer += 1
    dm["model.norm"] = len(split) - 1
    dm["model.rotary_emb"] = len(split) - 1
    return dm


class PipelinePrefill:
    """--pipeline-prefill: greedy()'s chunked prefill pipelined over the GPUs of a --device-map split placement."""

    def __init__(self, model, split):
        from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
        self.model = model
        inner = model.model
        cfg = inner.config
        if type(model).__name__ != "Qwen3ForCausalLM":
            raise SystemExit(f"--pipeline-prefill reproduces Qwen3Model.forward only; got {type(model).__name__}")
        if cfg._attn_implementation != "flash_attention_2":
            raise SystemExit("--pipeline-prefill needs --attn flash_attention_2")
        if set(cfg.layer_types) != {"full_attention"} or getattr(cfg, "is_causal", True) is not True:
            raise SystemExit("--pipeline-prefill: every layer must be causal full attention")
        lazy_import_flash_attention(cfg._attn_implementation)
        self.nl = cfg.num_hidden_layers
        self.devices = [torch.device("cuda", g) for g in range(len(split))]
        b = [0]
        for n in split:
            b.append(b[-1] + n)
        self.lo, self.hi = b[:-1], b[1:]
        for i, d in enumerate(self.devices):            # the loaded placement is the split
            for layer in range(self.lo[i], self.hi[i]):
                w = inner.layers[layer].input_layernorm.weight
                if w.device != d:
                    raise RuntimeError(f"layer {layer} on {w.device}, split wants {d}")
        self.last = {}

    def prefill(self, ids, chunk):
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from transformers.models.qwen3.modeling_qwen3 import create_causal_mask
        m = self.model
        inner = m.model
        cfg = inner.config
        L = int(ids.shape[1])
        bounds = [(s, min(L, s + chunk)) for s in range(0, L, chunk)]
        ng = len(self.devices)
        self.last = {}
        cache = DynamicCache(config=inner.config)
        if len(cache.layers) != self.nl or cache.layer_class_to_replicate is not None:
            raise RuntimeError("DynamicCache(config) did not pre-create one cache layer per decoder layer")
        pos, pe = [], []
        for s, e in bounds:
            x = inner.embed_tokens(ids[:, s:e])
            p = torch.arange(x.shape[1], device=x.device) + s
            p = p.unsqueeze(0)
            cos, sin = inner.rotary_emb(x, p)
            pos.append([p.to(d) for d in self.devices])
            pe.append([(cos.to(d), sin.to(d)) for d in self.devices])
            del x, cos, sin
        for d in self.devices:
            torch.cuda.synchronize(d)
        res = {}
        busy = [0.0] * ng
        hand = [0.0] * ng

        def stage(i, t, x):
            s, e = bounds[t]
            t0 = time.perf_counter()
            if i == 0:                                 # = Qwen3Model.forward up to the layer loop
                x = inner.embed_tokens(ids[:, s:e])
                seen = cache.get_seq_length()
                if seen != s:
                    raise RuntimeError(f"chunk {t}: cache holds {seen} tokens, expected {s}")
                mask = create_causal_mask(config=cfg, inputs_embeds=x, attention_mask=None,
                                          past_key_values=cache, position_ids=pos[t][0])
                if mask is not None:
                    raise RuntimeError("--pipeline-prefill: create_causal_mask returned a mask (FA2 without padding "
                                       "gives None)")
            for layer in range(self.lo[i], self.hi[i]):
                x = inner.layers[layer](x, attention_mask=None, position_embeddings=pe[t][i],
                                        position_ids=pos[t][i], past_key_values=cache, use_cache=True)
            if i == ng - 1:
                torch.cuda.synchronize(self.devices[i])
            busy[i] += time.perf_counter() - t0
            return x

        qs = [queue.Queue() for _ in range(ng)]
        errs = []
        stop = threading.Event()

        def run(i):
            try:
                torch.cuda.set_device(self.devices[i])
                with torch.inference_mode():           # inference / grad mode is per thread
                    while not stop.is_set():
                        item = qs[i].get()
                        if item is None:
                            break
                        t, xh = item
                        th = time.perf_counter()
                        x = None if xh is None else xh.to(self.devices[i])
                        hand[i] += time.perf_counter() - th
                        x = stage(i, t, x)
                        if i + 1 < ng:
                            th = time.perf_counter()
                            xh = x.to("cpu")            # synchronous D2H of the final x of this stage
                            hand[i] += time.perf_counter() - th
                            qs[i + 1].put((t, xh))
                        elif t == len(bounds) - 1:
                            res["x"] = x
                if i + 1 < ng:
                    qs[i + 1].put(None)
            except BaseException as exc:  # noqa: BLE001
                errs.append((i, exc))
                stop.set()
                for q_ in qs:
                    q_.put(None)

        ths = [threading.Thread(target=run, args=(i,), name=f"dense-stage{i}", daemon=True) for i in range(ng)]
        for th in ths:
            th.start()
        for t in range(len(bounds)):
            qs[0].put((t, None))
        qs[0].put(None)
        for th in ths:
            th.join()
        if errs:
            raise errs[0][1]
        if "x" not in res:
            raise RuntimeError("pipelined prefill: the last stage returned no hidden state")
        del pos, pe
        h = inner.norm(res["x"])                       # Qwen3Model.forward's final norm (whole last chunk)
        logits = m.lm_head(h[:, slice(-1, None), :])   # Qwen3ForCausalLM.forward, logits_to_keep=1
        self.last = dict(prefill_stage_busy_s=[round(v, 1) for v in busy],
                         prefill_handoff_s=[round(v, 2) for v in hand])
        return CausalLMOutputWithPast(logits=logits, past_key_values=cache)


@torch.inference_mode()
def greedy(model, ids, max_new, eos, prefill_chunk=0, timing=None, pipe=None):
    """Explicit greedy loop with an HF KV cache."""
    t_pf = time.time()
    if pipe is not None:
        out = pipe.prefill(ids, prefill_chunk)
    elif prefill_chunk > 0:
        past = None
        for s in range(0, ids.shape[1], prefill_chunk):
            out = model(ids[:, s:s + prefill_chunk], past_key_values=past,
                        use_cache=True, logits_to_keep=1)
            past = out.past_key_values
    else:
        out = model(ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    logits = out.logits[:, -1].float()
    if timing is not None:
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)
        timing["prefill_s"] = round(time.time() - t_pf, 1)
    got = []
    tok = torch.empty(1, 1, dtype=torch.long, device=ids.device)
    for i in range(max_new):
        nxt = int(logits.argmax(-1).item())
        got.append(nxt)
        if nxt in eos or i == max_new - 1:
            break
        tok.fill_(nxt)
        out = model(tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1].float()
    del past
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["student", "teacher"])
    ap.add_argument("--model", default=None, help="override the arm's default path")
    ap.add_argument("--chat-template", action="store_true")
    ap.add_argument("--suite", required=True, help="LEN=path/to/eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--mnt-override", type=int, default=0,
                    help="force max_new_tokens for EVERY row (0 = off); the "
                         "instrument fix, same flag as the RecSA "
                         "driver")
    ap.add_argument("--extra-eos", type=int, nargs="*", default=[],
                    help="additional stop token ids (e.g. 151643)")
    ap.add_argument("--rope-scaling-json", default=None,
                    help='e.g. {"rope_type":"yarn","factor":2.0,'
                         '"original_max_position_embeddings":262144} — '
                         "minimal-sufficient-factor extension; merged into "
                         "rope_parameters, never assigned wholesale (see "
                         "stage0_teacher_nscand)")
    ap.add_argument("--device-map", default="none", choices=["none", "auto", "split"],
                    help="none (default) = model.to('cuda'); auto = HF/accelerate "
                         "device_map='auto' over the visible GPUs (1M: KV > 1 GPU); "
                         "split = an explicit contiguous layer split (--layer-split), "
                         "the other modules where 'auto' puts them")
    ap.add_argument("--layer-split", default=None,
                    help="with --device-map split: decoder layers per visible GPU, "
                         "e.g. 9,9,9,9 (default: an even split over the visible GPUs)")
    ap.add_argument("--pipeline-prefill", action="store_true",
                    help="with --device-map split, --prefill-chunk > 0 and "
                         "--attn flash_attention_2: run the chunked prefill as a "
                         "stage pipeline, one thread per GPU")
    ap.add_argument("--max-memory-gib", type=float, default=0,
                    help="with --device-map auto: per-visible-GPU weight budget "
                         "in GiB for the placement (0 = accelerate's balanced default)")
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="0 (default) = one-shot prefill; N > 0 = feed the prompt "
                         "in N-token chunks through past_key_values (use with "
                         "--attn flash_attention_2 at long L)")
    args = ap.parse_args()
    if args.max_memory_gib and args.device_map != "auto":
        raise SystemExit("--max-memory-gib needs --device-map auto")
    if args.prefill_chunk < 0:
        raise SystemExit("--prefill-chunk must be >= 0")
    if args.layer_split is not None and args.device_map != "split":
        raise SystemExit("--layer-split needs --device-map split")
    if args.pipeline_prefill and (args.device_map != "split" or args.prefill_chunk <= 0
                                  or args.attn != "flash_attention_2"):
        raise SystemExit("--pipeline-prefill needs --device-map split, --prefill-chunk > 0 "
                         "and --attn flash_attention_2")
    long_mode = (args.device_map != "none" or args.prefill_chunk > 0
                 or bool(args.max_memory_gib))

    default = {
        "student": __import__("os").environ.get("RECSA_EXPORT", ""),
        "teacher": "Qwen/Qwen3-4B",
    }
    model_id = args.model or default[args.arm]

    eval_len, _, suite_path = args.suite.partition("=")
    eval_len = int(eval_len)
    rows = [json.loads(l) for l in open(suite_path)]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_sample = out / "per_sample.jsonl"
    done = set()
    if per_sample.exists():
        for l in open(per_sample):
            try:
                done.add(json.loads(l)["sample_id"])
            except json.JSONDecodeError:
                pass  # tolerate a truncated last line from a killed run

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    cfg = AutoConfig.from_pretrained(model_id, local_files_only=True)
    if args.rope_scaling_json:
        override = json.loads(args.rope_scaling_json)
        existing = dict(getattr(cfg, "rope_parameters", None) or {})
        merged = {**existing, **override}
        if merged.get("rope_theta") is None:
            theta = existing.get("rope_theta", getattr(cfg, "rope_theta", None))
            if theta is not None:
                merged["rope_theta"] = theta
        cfg.rope_scaling = merged
        factor = float(override.get("factor", 1.0))
        omp = int(override.get("original_max_position_embeddings",
                               cfg.max_position_embeddings))
        cfg.max_position_embeddings = max(cfg.max_position_embeddings,
                                          int(omp * factor))
    need = max(int(r["actual_tokens"])
               + (args.mnt_override or int(r["metadata"]["max_new_tokens"]))
               for r in rows) + 256
    if cfg.max_position_embeddings < need:
        raise SystemExit(
            f"{model_id} window {cfg.max_position_embeddings} < {need} needed. This "
            f"driver refuses to silently extrapolate: a dense arm past its window "
            f"measures unscaled-RoPE failure, not attention.")
    split = pipe = None
    if args.device_map == "none":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=cfg, torch_dtype=torch.bfloat16, local_files_only=True,
            attn_implementation=args.attn).to("cuda").eval()
        in_dev = "cuda"
        peak_devs = None
    else:
        mm = ({i: f"{args.max_memory_gib}GiB" for i in range(torch.cuda.device_count())}
              if args.max_memory_gib else None)
        if args.device_map == "split":
            ng, nl = torch.cuda.device_count(), cfg.num_hidden_layers
            split = ([int(v) for v in args.layer_split.split(",")] if args.layer_split
                     else [nl // ng + (1 if g < nl % ng else 0) for g in range(ng)])
            if len(split) != ng:
                raise SystemExit(f"--layer-split {split}: {ng} visible GPUs")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=cfg, torch_dtype=torch.bfloat16, local_files_only=True,
            attn_implementation=args.attn,
            device_map=("auto" if split is None else split_device_map(cfg.num_hidden_layers, split)),
            max_memory=mm).eval()
        in_dev = model.get_input_embeddings().weight.device
        peak_devs = list(range(torch.cuda.device_count()))
        dm = getattr(model, "hf_device_map", None) or {}
        dm_layers = {}
        for k, v in dm.items():
            n = (cfg.num_hidden_layers if k == "" else
                 1 if re.search(r"(^|\.)layers\.\d+$", k) else 0)
            if n:
                dm_layers[str(v)] = dm_layers.get(str(v), 0) + n
        print(f"[device-map] input on {in_dev}; decoder layers per device {dm_layers}; "
              f"lm_head on {model.get_output_embeddings().weight.device}; "
              f"visible GPUs {peak_devs}", flush=True)
        if split is not None:
            print(f"[device-map] split {split}: " + json.dumps(
                {k: v for k, v in dm.items() if "layers." not in k}), flush=True)
        if args.pipeline_prefill:
            pipe = PipelinePrefill(model, split)
            print(f"[pipeline-prefill] {len(split)} stages, layers {split}, "
                  f"steps = the {args.prefill_chunk}-token prefill chunks", flush=True)
    if args.prefill_chunk > 0 and args.attn == "sdpa":
        print("[prefill-chunk] note: sdpa materialises a (chunk x kv_len) mask "
              "per chunk with a past; use --attn flash_attention_2 at long L",
              flush=True)
    eos = {int(t) for t in (cfg.eos_token_id if isinstance(cfg.eos_token_id, list)
                            else [cfg.eos_token_id]) if t is not None}
    eos.update(int(t) for t in args.extra_eos)
    theta_rec = env_assert.check_rope_theta(model.config, what="dense arm rope_theta")
    chat_kw = env_assert.chat_template_kwargs(tok) if args.chat_template else {}

    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    t_start = time.time()
    n_done = n_err = 0
    with open(per_sample, "a") as fout:
        for r in rows:
            if r["sample_id"] in done:
                n_done += 1
                continue
            mnt = args.mnt_override or int(r["metadata"]["max_new_tokens"])
            golds = r.get("answers") or [r["answer"]]
            match = r["metadata"].get("match", "all")
            if peak_devs is None:
                torch.cuda.reset_peak_memory_stats()
            else:
                for d in peak_devs:
                    torch.cuda.reset_peak_memory_stats(d)
            timing = {} if long_mode else None
            t0 = time.time()
            try:
                text = r["text"]
                if args.chat_template:
                    text = tok.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False, add_generation_prompt=True, **chat_kw)
                ids = tok(text, add_special_tokens=False,
                          return_tensors="pt").input_ids.to(in_dev)
                if pipe is not None:
                    got = greedy(model, ids, mnt, eos,
                                 prefill_chunk=args.prefill_chunk, timing=timing, pipe=pipe)
                elif long_mode:
                    got = greedy(model, ids, mnt, eos,
                                 prefill_chunk=args.prefill_chunk, timing=timing)
                else:
                    got = greedy(model, ids, mnt, eos)
                pred = tok.decode(got, skip_special_tokens=True)
                peak_dev = ([round(torch.cuda.max_memory_allocated(d) / 2 ** 30, 2)
                             for d in peak_devs] if peak_devs is not None else None)
                row = dict(
                    eval_len=eval_len, sample_id=r["sample_id"],
                    task_type=r["task_type"], arm=f"dense_{args.arm}",
                    chat_template=bool(args.chat_template),
                    position_bucket=r.get("position_bucket", "na"),
                    score=score_row(pred, golds, match),
                    score_bnd=score_bnd(pred, golds, match),
                    score_tok=score_row_tok(pred, golds, match),
                    score_tok_any=score_row_tok_any(pred, golds, match),
                    score_tok_credited=score_row_tok_credited(
                        pred, golds, match, score_row_tok(pred, golds, match)),
                    em=float(normalize_prediction(pred)
                             == normalize_prediction(str(golds[0]))),
                    em_fixed=float(normalize_prediction(pred).strip()
                                   == normalize_prediction(str(golds[0])).strip()),
                    n_golds=len(golds), prefill_tokens=int(ids.shape[1]),
                    mnt_used=mnt,
                    gen_tokens=len(got), wall_s=round(time.time() - t0, 1),
                    peak_gib=(round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                              if peak_dev is None else max(peak_dev)),
                    pred=pred[:300], pred_full=pred)
                if peak_dev is not None:
                    row["peak_gib_per_device"] = peak_dev
                if long_mode:
                    row["prefill_s"] = timing.get("prefill_s")
                if pipe is not None:
                    row.update(pipe.last)
            except BaseException as e:  # noqa: BLE001
                if isinstance(e, KeyboardInterrupt):
                    raise
                n_err += 1
                row = dict(eval_len=eval_len, sample_id=r["sample_id"],
                           task_type=r["task_type"], arm=f"dense_{args.arm}",
                           error=repr(e)[:300], tb=traceback.format_exc()[-800:],
                           wall_s=round(time.time() - t0, 1))
                print(f"[ERROR] {r['sample_id']}: {e!r}", flush=True)
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_done += 1
            s = row.get("score")
            print(f"[{n_done}/{len(rows)}] {r['task_type']} "
                  f"score={'ERR' if s is None else f'{s:.2f}'} "
                  f"{row['wall_s']:.0f}s", flush=True)

    all_rows = []
    for l in open(per_sample):
        try:
            all_rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    good = [r for r in all_rows if "score" in r]
    errs = [r for r in all_rows if "error" in r]
    by_id = {r["sample_id"]: r for r in rows}
    kept = []
    for r in good:
        if "score_tok" in r and "score_tok_any" in r:
            kept.append(r)
            continue
        src = by_id.get(r["sample_id"])
        if src is None:
            print(f"[skip] {r['sample_id']} predates score_tok and is outside "
                  f"the current suite slice — excluded from the aggregate",
                  flush=True)
            continue
        g_, m_ = (src.get("answers") or [src["answer"]]), \
            src["metadata"].get("match", "all")
        r["score_tok"] = score_row_tok(r["pred_full"], g_, m_)
        r["score_tok_any"] = score_row_tok_any(r["pred_full"], g_, m_)
        r["score_tok_credited"] = score_row_tok_credited(
            r["pred_full"], g_, m_, r["score_tok"])
        kept.append(r)
    good = kept
    if not good:
        raise SystemExit(f"DENSE-RULER-FAILED 0 scored rows, {len(errs)} errors")
    by_task = {}
    for r in good:
        by_task.setdefault(r["task_type"], []).append(r)
    agg = lambda rs, k: round(sum(r[k] for r in rs) / len(rs), 4)  # noqa: E731
    summary = dict(
        model=model_id, arm=f"dense_{args.arm}", attn=args.attn,
        chat_template=bool(args.chat_template), graft=None,
        decode="dense_hf_kvcache_greedy", git_rev=rev,
        window=cfg.max_position_embeddings,
        rope_theta=theta_rec["rope_theta"],
        rope_theta_resolution=theta_rec["rope_theta_resolution"],
        instrument=dict(mnt_override=args.mnt_override,
                        extra_eos=list(args.extra_eos),
                        rope_scaling=args.rope_scaling_json,
                        chat_template_kwargs=chat_kw or None,
                        chat_template_hybrid=(env_assert.hybrid_thinking_template(tok)
                                              if args.chat_template else None)),
        suites=[dict(len=eval_len, path=suite_path, sha256=sha256(suite_path))],
        panel={str(eval_len): dict(
            peak_gib=round(max(r.get("peak_gib", 0) for r in good), 2),
            wall_s=round(time.time() - t_start, 1), n_error=len(errs),
            all_task=dict(string_match=agg(good, "score"),
                          string_match_bnd=agg(good, "score_bnd"),
                          string_match_tok=agg(good, "score_tok"),
                          string_match_tok_any=agg(good, "score_tok_any"),
                          em=agg(good, "em"), em_fixed=agg(good, "em_fixed"),
                          n=len(good)),
            by_task={t: dict(string_match=agg(rs, "score"),
                             string_match_bnd=agg(rs, "score_bnd"),
                             string_match_tok=agg(rs, "score_tok"),
                             string_match_tok_any=agg(rs, "score_tok_any"),
                             em=agg(rs, "em"), n=len(rs))
                     for t, rs in sorted(by_task.items())})})
    if long_mode:
        summary["instrument"].update(
            device_map=args.device_map, prefill_chunk=args.prefill_chunk,
            max_memory_gib=args.max_memory_gib or None,
            device_map_layers=(dm_layers if args.device_map != "none" else None))
    if split is not None:
        summary["instrument"].update(layer_split=split,
                                     pipeline_prefill=bool(args.pipeline_prefill))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"PEAK-MEM {summary['panel'][str(eval_len)]['peak_gib']} GiB")
    if errs:
        print(f"N-ERROR {len(errs)}")
    print("DENSE-RULER-DONE")


if __name__ == "__main__":
    main()
