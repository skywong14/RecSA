"""Generative RULER eval over the RecSA decode path."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from recsa.ruler.rulerlite import normalize_prediction  # noqa: E402
from recsa.decode import RecSADecodeSession  # noqa: E402
from recsa.leaf_cuda import set_cuda_strict  # noqa: E402
from recsa.parallel import recsa_env, prefill_config  # noqa: E402
from recsa.probe_pool import compute_num_layers  # noqa: E402
from graft import install_recsa_attention, install_streamed_forward  # noqa: E402
import env_assert  # noqa: E402
import graft_geometry  # noqa: E402


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                return h.hexdigest()
            h.update(b)


def score_row(pred: str, golds, match: str):
    p = pred.lower()
    hits = [1.0 if g.lower().strip() in p else 0.0 for g in golds]
    return max(hits) if match == "part" else sum(hits) / len(hits)


def score_row_bnd(pred: str, golds, match: str):
    p = pred.lower()
    hits = [1.0 if re.search(r"(?<![0-9a-z_])" + re.escape(g.lower().strip())
                             + r"(?![0-9a-z_])", p) else 0.0 for g in golds]
    return max(hits) if match == "part" else sum(hits) / len(hits)


def extract_answer_token(pred: str) -> str:
    line = normalize_prediction(pred)
    ms = list(re.finditer(r"\b(?:answer|is|are)\b\s*[:\-]?\s*", line))
    if ms:
        m = re.search(r"[a-z0-9]+", line[ms[-1].end():])
        return m.group(0) if m else ""
    return line


def score_row_tok(pred: str, golds, match: str):
    t = extract_answer_token(pred)
    hits = [1.0 if t == normalize_prediction(str(g)) else 0.0 for g in golds]
    return max(hits) if match == "part" else sum(hits) / len(hits)


def score_row_tok_any(pred: str, golds, match: str):
    """Any-line variant of score_row_tok (normalize_prediction reads only the first line)."""
    lines = [ln for ln in pred.splitlines() if ln.strip()] or [pred]
    hits = []
    for g in golds:
        gn = normalize_prediction(str(g))
        hits.append(1.0 if any(extract_answer_token(ln) == gn for ln in lines)
                    else 0.0)
    return max(hits) if match == "part" else sum(hits) / len(hits)


_CRED_RE = re.compile(r"(?:answer|is|are)\s*[:\s]\s*\**([A-Za-z0-9]+)\b", re.I)


def score_row_tok_credited(pred: str, golds, match: str, base: float):
    if base == 1.0:
        return 1.0
    caps = _CRED_RE.findall(pred)
    if not caps:
        return base
    hits = []
    for g in golds:
        gn = normalize_prediction(str(g))
        hits.append(1.0 if any(normalize_prediction(c) == gn for c in caps)
                    else 0.0)
    v = max(hits) if match == "part" else sum(hits) / len(hits)
    return max(base, v)


# files hashed into the run provenance
SRC_PIN = ["recsa/parallel.py", "recsa/incremental_tree.py", "recsa/decode.py",
           "recsa/probe_pool.py", "scripts/graft.py",
           "recsa/true_rope.py",
           "scripts/rope_scaling.py",
           "scripts/ruler_eval.py",
           "scripts/graft_geometry.py",
           "scripts/env_assert.py",
           "scripts/decode_sel_trace.py",
           "recsa/merge_acc.py", "recsa/fast_select.py", "recsa/leaf_cuda.py", "recsa/topsel_cuda.py",
           "recsa/pool_fused.py"] + sorted(str(p.relative_to(REPO)) for p in (REPO / "recsa" / "cuda").glob("*.cu"))


def decode_label(offload_leaves: bool, graph: bool, summaries: str,
                 summary_norm=None, graft_mode: str = "internal", rope_scaling=None) -> str:
    """summary.json 'decode' label."""
    lab = (f"recsa_phase{'15' if offload_leaves else '1'}"
           f"_{'graph' if graph else 'eager'}_greedy")
    lab += "_nosum" if summaries == "none" else ""
    if summary_norm is not None:
        lab += "_" + graft_geometry.summary_norm_tag(summary_norm)
    if graft_mode == "external":
        lab += "_ropeext"
        if rope_scaling is not None:
            f = float(rope_scaling.get("factor", 1.0))
            lab += f"_{rope_scaling.get('rope_type', 'rs')}{int(f) if f.is_integer() else f}"
    return lab


def resolve_run_geometry(model_dir, top_k, summaries, log=print, summary_norm=None) -> dict:
    geom = graft_geometry.resolve(model_dir, top_k=top_k, summaries=summaries, log=log,
                                  summary_norm=summary_norm)
    log(f"[graft] summaries={geom['summaries']} (recorded="
        f"{(geom['recorded'] or {}).get('summaries')}, off_config={geom['off_config']})")
    log(f"[graft] summary_norm={geom['summary_norm']} (recorded="
        f"{(geom['recorded'] or {}).get('summary_norm')}, off_config={geom['off_config']})")
    return geom


def install_run_graft(model, geom, *, mode, rope_base, probe_ckpt, q_chunk,
                      build_chunk_parents):
    install_recsa_attention(model, mode=mode,
                            rope_base=rope_base,
                            probe_ckpt=probe_ckpt,
                            compression_rate=geom["compression_rate"],
                            top_k=geom["top_k"],
                            max_top_nodes=geom["max_top_nodes"],
                            q_chunk=q_chunk,
                            build_chunk_parents=build_chunk_parents,
                            summaries=geom["summaries"],
                            summary_norm=geom.get("summary_norm"))
    install_streamed_forward(model, q_chunk=q_chunk)
    acfg = model.model.layers[0].self_attn._recsa_cfg
    for i, layer in enumerate(model.model.layers):
        c = getattr(layer.self_attn, "_recsa_cfg", None)
        if c is not acfg or c.get("summaries") != geom["summaries"]:
            raise RuntimeError(
                f"layer {i} graft cfg runs summaries="
                f"{(c or {}).get('summaries')!r} (shared cfg: {c is acfg}) but "
                f"the resolved geometry says {geom['summaries']!r} — refusing "
                "to record one operator and run another")
        if c.get("summary_norm") != geom.get("summary_norm"):
            raise RuntimeError(
                f"layer {i} graft cfg runs summary_norm={c.get('summary_norm')!r} but "
                f"the resolved geometry says {geom.get('summary_norm')!r} — refusing "
                "to record one operator and run another")
    return acfg


def resume_check_summary_norm(per_sample, summary_json, summary_norm) -> None:
    """Refuse to resume an out dir written with another summary_norm."""
    summary_json = Path(summary_json)
    if summary_json.is_file():
        try:
            prev = json.loads(summary_json.read_text()).get("summary_norm")
            readable = True
        except json.JSONDecodeError:
            readable = False
        if readable and prev != summary_norm:
            raise SystemExit(
                f"{summary_json} records summary_norm={prev!r}; this run is "
                f"summary_norm={summary_norm!r} — refusing to resume one operator "
                "variant's cell with the other (use a fresh --out)")
    if Path(per_sample).exists():
        for l in open(per_sample):
            try:
                rr = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "score" in rr and rr.get("summary_norm") != summary_norm:
                raise SystemExit(
                    f"{per_sample}: row {rr.get('sample_id')} was scored with "
                    f"summary_norm={rr.get('summary_norm')!r}; this run is "
                    f"summary_norm={summary_norm!r} — refusing to mix operator "
                    "variants in one cell (use a fresh --out)")


def resume_check_graft(per_sample, summary_json, graft_mode, rope_scaling) -> None:
    """Refuse to resume an out dir written with another (graft_mode, rope_scaling)."""
    summary_json = Path(summary_json)
    want = (graft_mode, rope_scaling)
    if summary_json.is_file():
        try:
            prev = json.loads(summary_json.read_text())
            got = (prev.get("graft_mode", "internal"), prev.get("rope_scaling"))
        except json.JSONDecodeError:
            got = None
        if got is not None and got != want:
            raise SystemExit(f"{summary_json} records (graft_mode, rope_scaling)={got!r}; this run is {want!r} — "
                             "refusing to resume one RoPE variant's cell with another (use a fresh --out)")
    if Path(per_sample).exists():
        for l in open(per_sample):
            try:
                rr = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "score" in rr and (rr.get("graft_mode", "internal"), rr.get("rope_scaling")) != want:
                raise SystemExit(f"{per_sample}: row {rr.get('sample_id')} was scored with (graft_mode, "
                                 f"rope_scaling)=({rr.get('graft_mode', 'internal')!r}, {rr.get('rope_scaling')!r}); "
                                 f"this run is {want!r} — refusing to mix RoPE variants in one cell (use a fresh --out)")


def parse_rope_scaling(s):
    """--rope-scaling-json: a JSON object (e.g. {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings":
    32768}), canonicalised (sorted keys); None when not given."""
    if s is None:
        return None
    try:
        d = json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--rope-scaling-json is not JSON: {e}") from None
    if not isinstance(d, dict) or not d.get("rope_type"):
        raise argparse.ArgumentTypeError("--rope-scaling-json must be an object with a rope_type")
    return dict(sorted(d.items()))


def apply_rope_scaling(cfg, override: dict) -> None:
    import rope_scaling
    rope_scaling.merge_rope_scaling(cfg, override)


def resume_done_ids(per_sample, summary_json, summaries) -> set:
    """sample_ids already scored in an existing out dir. Error rows are retried."""
    summary_json = Path(summary_json)
    if summary_json.is_file():
        try:
            prev = json.loads(summary_json.read_text()).get("summaries", "softmax")
        except json.JSONDecodeError:
            prev = None
        if prev is not None and prev != summaries:
            raise SystemExit(
                f"{summary_json} records summaries={prev!r}; this run is "
                f"summaries={summaries!r} — refusing to resume one operator "
                "variant's cell with the other (use a fresh --out)")
    done = set()
    if Path(per_sample).exists():
        for l in open(per_sample):
            try:
                rr = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "score" in rr:
                if rr.get("summaries", "softmax") != summaries:
                    raise SystemExit(
                        f"{per_sample}: row {rr.get('sample_id')} was scored "
                        f"with summaries={rr.get('summaries', 'softmax')!r}; "
                        f"this run is summaries={summaries!r} — refusing to "
                        "mix operator variants in one cell (use a fresh --out)")
                done.add(rr["sample_id"])
    return done


def provenance(probe_ckpt=None):
    try:
        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10
                             ).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10
                               ).stdout.strip()
    except Exception:
        rev, dirty = "", ""
    p = dict(git_rev=rev or "unavailable", git_dirty=bool(dirty),
             files={f: sha256(REPO / f) for f in SRC_PIN})
    if probe_ckpt:
        p["probe_ckpt_sha256"] = sha256(probe_ckpt)
    p["recsa_env"] = recsa_env()
    p["prefill"] = prefill_config()
    return p


def force_nodes_per_level(spans_tok, num_layers, c, prompt_len, arm, seed_key):
    """Nodes to force-select, {level: sorted node ids} for levels NL-1..1 (leaf level has no selection); evidence =
    every node covering a span."""
    import random as _random
    rng = _random.Random(seed_key)

    def nodes_for(spans):
        return {li: sorted({p for ts, te in spans
                            for p in range(ts // (c ** li),
                                           (te - 1) // (c ** li) + 1)})
                for li in range(num_layers - 1, 0, -1)}

    ev_out = nodes_for(spans_tok)
    if arm == "evidence":
        return ev_out

    quantum = c ** (num_layers - 1)
    lo_t = min(ts for ts, _ in spans_tok)
    hi_t = max(te for _, te in spans_tok)
    usable = max(prompt_len - 1024, hi_t)
    ev_sets = {li: set(v) for li, v in ev_out.items()}
    m_lo, m_hi = -(lo_t // quantum), (usable - hi_t) // quantum
    for _ in range(4000):
        if m_hi <= m_lo:
            break
        m = rng.randint(m_lo, m_hi)
        if m == 0:
            continue
        delta = m * quantum
        cand = nodes_for([(ts + delta, te + delta) for ts, te in spans_tok])
        if any(set(cand[li]) & ev_sets[li] for li in cand):
            continue
        assert all(len(cand[li]) == len(ev_out[li]) for li in cand), \
            f"relocation changed per-level counts: {cand} vs {ev_out}"
        return cand
    raise RuntimeError(
        f"no valid random-arm relocation for spans={spans_tok} "
        f"prompt_len={prompt_len} quantum={quantum}")


def force_bitmasks(fnodes, cap_len, c, top_k, device):
    """Per-level int32 bitmask tensors sized to the session capacity."""
    masks = {}
    for li, nodes in fnodes.items():
        if not nodes:
            continue
        assert len(nodes) + 1 <= top_k, (
            f"level {li}: {len(nodes)} forced nodes + rightmost exceeds "
            f"top_k={top_k}")
        lp = c ** li
        nwords = (cap_len // lp + c + 32) // 32 + 1
        wl = [0] * nwords
        for n in nodes:
            wl[n >> 5] |= 1 << (n & 31)
        wl = [w - (1 << 32) if w >= (1 << 31) else w for w in wl]
        masks[li] = torch.tensor(wl, dtype=torch.int32, device=device)
    return masks


def wrap_decode_scope(sess, acfg, fmasks):
    """--force-scope=decode: masks off for the prefill, on for every decode step."""
    base_prefill = sess.prefill

    def prefill(input_ids):
        acfg.pop("force_masks", None)
        out = base_prefill(input_ids)
        acfg["force_masks"] = fmasks
        return out

    sess.prefill = prefill
    return sess


def wrap_decode_scope_by_layer(sess, acfg, fmbl):
    base_prefill = sess.prefill

    def prefill(input_ids):
        acfg.pop("force_masks_by_layer", None)
        out = base_prefill(input_ids)
        acfg["force_masks_by_layer"] = fmbl
        return out

    sess.prefill = prefill
    return sess


def wrap_decode_bias(sess, prep):
    import types
    base_prefill = sess.prefill

    def prefill(input_ids):
        out = base_prefill(input_ids)
        for li, st in enumerate(sess.states):
            d = prep["layers"][str(li)]
            assert getattr(st, "_g_b", 0.0) == 0.0, \
                "gate_b is live — bias-arm would collide with the trained bias"
            bufs = {}
            for lvl, key in ((1, "bias_mid"), (2, "bias_top")):
                b = d[key].float()
                cap = st._bias_cap(lvl)
                buf = torch.zeros(1, cap, b.shape[1], dtype=torch.float32,
                                  device=st._k[lvl].device)
                n = min(b.shape[0], cap)
                buf[0, :n] = b[:n].to(buf.device)
                bufs[lvl] = buf
            st._b3_bufs = bufs

            def _lb(self):
                return [None] + [self._b3_bufs[l][:, : self._sizes[l]]
                                 for l in range(1, len(self._sizes))]
            st.layers_bias = types.MethodType(_lb, st)
        return out

    sess.prefill = prefill
    return sess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--graft", default="internal:auto",
                    help="internal:auto = the model's own rope_theta (a model property: "
                         "1e6 on the original Qwen3-4B); internal:<theta> must equal it (env_assert)")
    ap.add_argument("--top-k", type=int, default=None,
                    choices=graft_geometry.ALLOWED_TOP_K,
                    help=f"beam width; default = trained value "
                         f"({graft_geometry.DEFAULT_TOP_K}). Off-config values "
                         "are permitted and recorded in the summary.")
    ap.add_argument("--summaries", default=None,
                    choices=graft_geometry.SUMMARIES,
                    help="softmax = default operator, none = no pooled summaries in the "
                         "softmax; default = the export's recorded value. Off-config values "
                         "are permitted and recorded in the summary.")
    ap.add_argument("--summary-norm", default=None, type=graft_geometry.parse_summary_norm,
                    help="none | mass_cap:<tau>[:n_ref=<int>] (mass_cap caps each query "
                         "head's summed summary probability at tau, 0 < tau < 1, with n_ref only "
                         "on rows whose summary count exceeds it; the other rules are "
                         "not part of this package); default = "
                         "the export's recorded value (absent = off). Off-config values are "
                         "permitted and recorded in the summary.")
    ap.add_argument("--rope-scaling-json", default=None, type=parse_rope_scaling,
                    help='--graft external only: e.g. {"rope_type":"yarn","factor":4.0,'
                         '"original_max_position_embeddings":32768}, merged into the config\'s '
                         "rope_parameters before the model is built (as ruler_eval_dense)")
    ap.add_argument("--resolve-only", action="store_true",
                    help="print the resolved graft geometry / summary_norm / decode label as "
                         "JSON and exit (no suite read, no model load)")
    ap.add_argument("--q-chunk", type=int, default=131072)
    ap.add_argument("--probe-ckpt", default=None)
    ap.add_argument("--suite", required=True, help="LEN=path/to/eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reserve-pad", type=int, default=8)
    ap.add_argument("--offload-leaves", action="store_true",
                    help="leaf-offload session: leaves in a registered host pool"
                         "; bitwise equal to the all-GPU session")
    ap.add_argument("--graph", action="store_true",
                    help="CUDA-graph decode steps")
    ap.add_argument("--build-chunk-parents", type=int, default=None,
                    help="tiled probe build: REQUIRED >5M; only "
                         "cert-covered (slice, full-level) pairs are legal")
    ap.add_argument("--mnt-default", type=int, default=64,
                    help="max_new_tokens when the suite row carries none "
                         "(nscand-era suites)")
    ap.add_argument("--mnt-override", type=int, default=0,
                    help="force max_new_tokens for every row (0 = off)")
    ap.add_argument("--extra-eos", type=int, nargs="*", default=[],
                    help="additional stop token ids on top of the model "
                         "config's eos set (e.g. 151643)")
    ap.add_argument("--chat-template", action="store_true",
                    help="wrap row text as a single user turn via the "
                         "tokenizer chat template (add_generation_prompt)")
    ap.add_argument("--trace-selection", action="store_true",
                    help="per-step reference-descent selection trace "
                         "against the row's evidence_spans "
                         "(decode_sel_trace.py); eager decode only")
    ap.add_argument("--force-buckets",
                    choices=["off", "evidence", "random", "static"],
                    default="off",
                    help="oracle arms: force-include buckets in "
                         "the top-k candidate set at every level (evidence = "
                         "the row's evidence_spans; random = the same "
                         "per-level COUNT of non-evidence buckets, seeded per "
                         "sample). Budget stays exactly top_k — the forced "
                         "buckets displace the lowest-scoring survivors. "
                         "Eager decode only.")
    ap.add_argument("--force-scope", choices=["all", "decode"], default="all",
                    help="apply forcing to prefill+decode (all, the primary "
                         "oracle) or to decode steps only (decode, the "
                         "strictly weaker intervention)")
    ap.add_argument("--force-seed", type=int, default=0,
                    help="base seed for the random arm (mixed with sample_id)")
    ap.add_argument("--prep-dir", default=None,
                    help="prep dir (required for "
                         "--force-buckets=static and --bias-arm=cone); one "
                         "<shard_stem>.pt per row, keyed by suite file stem")
    ap.add_argument("--bias-arm", choices=["off", "cone"], default="off",
                    help="post-prefill per-(layer,level,node,kv) "
                         "cone-center bias via each state's layers_bias() "
                         "(the level_bias carrier). Decode scope by construction; "
                         "eager decode only.")
    args = ap.parse_args()
    set_cuda_strict(True)

    graft_mode_arg = args.graft.partition(":")[0]
    if args.rope_scaling_json is not None and graft_mode_arg != "external":
        ap.error("--rope-scaling-json needs --graft external (the internal graft's candidate-order RoPE "
                 "does not read the HF rotary embedding)")
    if graft_mode_arg == "external" and args.trace_selection:
        ap.error("--trace-selection replays internal (candidate-order) RoPE selection; not with --graft external")
    if args.trace_selection and args.graph:
        ap.error("--trace-selection hooks the eager step; drop --graph")
    if args.force_buckets != "off" and args.graph:
        ap.error("--force-buckets needs the eager select path; drop --graph")
    if args.bias_arm != "off" and args.graph:
        ap.error("--bias-arm needs the eager decode path (level_bias carrier); "
                 "drop --graph")
    if (args.force_buckets == "static" or args.bias_arm != "off") \
            and not args.prep_dir:
        ap.error("--force-buckets=static / --bias-arm=cone require --prep-dir")
    if args.force_buckets == "static" and args.force_scope != "decode":
        ap.error("--force-buckets=static is registered decode-scope only")
    if args.trace_selection or args.force_buckets != "off":
        from decode_sel_trace import spans_to_token_ranges
    if args.trace_selection:
        from decode_sel_trace import SelectionTrace, TracingSession

    geom = resolve_run_geometry(args.model, args.top_k, args.summaries,
                                summary_norm=args.summary_norm)
    if geom["summary_norm"] is not None and args.bias_arm != "off":
        ap.error("--bias-arm (level_bias carrier) is not combinable with a "
                 "summary_norm rule")
    if args.resolve_only:
        print(json.dumps(dict(graft_geometry=geom, summary_norm=geom["summary_norm"],
                              decode=decode_label(args.offload_leaves, args.graph,
                                                  geom["summaries"], geom["summary_norm"],
                                                  graft_mode_arg, args.rope_scaling_json))))
        return

    eval_len, _, suite_path = args.suite.partition("=")
    eval_len = int(eval_len)
    rows = [json.loads(l) for l in open(suite_path)]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_sample = out / "per_sample.jsonl"
    done = resume_done_ids(per_sample, out / "summary.json", geom["summaries"])
    resume_check_summary_norm(per_sample, out / "summary.json", geom["summary_norm"])
    resume_check_graft(per_sample, out / "summary.json", graft_mode_arg, args.rope_scaling_json)

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cfg = AutoConfig.from_pretrained(args.model)
    if args.rope_scaling_json is not None:
        apply_rope_scaling(cfg, args.rope_scaling_json)
    need = max(int(r["actual_tokens"])
               + int(r["metadata"].get("max_new_tokens", args.mnt_default))
               for r in rows) + 256
    cfg.max_position_embeddings = max(cfg.max_position_embeddings, need)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    mode, theta, theta_rec = env_assert.graft_rope_base(args.graft, model.config)
    rotary_rec = None
    if mode == "external":
        rot = model.model.rotary_emb
        print(f"[graft] RoPE: TRUE-POSITION (graft mode external; kernel rope_mode none); rotary "
              f"rope_type={rot.rope_type} attention_scaling={rot.attention_scaling} "
              f"rope_scaling={args.rope_scaling_json}", flush=True)
        import rope_scaling as _rs
        rotary_rec = _rs.check_rotary(model, args.rope_scaling_json, theta=theta)
    acfg = install_run_graft(model, geom, mode=mode, rope_base=theta,
                             probe_ckpt=args.probe_ckpt, q_chunk=args.q_chunk,
                             build_chunk_parents=args.build_chunk_parents)
    eos = [tid for tid in (model.config.eos_token_id
                           if isinstance(model.config.eos_token_id, list)
                           else [model.config.eos_token_id]) if tid is not None]
    for tid in args.extra_eos:
        if tid not in eos:
            eos.append(tid)

    prov = provenance(args.probe_ckpt)
    chat_kw = env_assert.chat_template_kwargs(tok) if args.chat_template else {}
    t_start = time.time()
    n_done = 0
    with open(per_sample, "a") as fout:
        for r in rows:
            if r["sample_id"] in done:
                n_done += 1
                continue
            torch.cuda.reset_peak_memory_stats()
            sess = None
            try:
                mnt = args.mnt_override or int(r["metadata"].get(
                    "max_new_tokens", args.mnt_default))
                raw = r["text"]
                if args.chat_template:
                    text = tok.apply_chat_template(
                        [{"role": "user", "content": raw}],
                        tokenize=False, add_generation_prompt=True, **chat_kw)
                else:
                    text = raw
                trace = None
                spans_char = []
                if args.trace_selection or args.force_buckets != "off":
                    spans_char = [(sp["start_char"], sp["end_char"])
                                  for sp in (r.get("evidence_spans") or [])
                                  if sp.get("start_char") is not None]
                    if args.force_buckets in ("evidence", "random") \
                            and not spans_char:
                        raise RuntimeError(
                            f"--force-buckets={args.force_buckets} but row "
                            f"{r['sample_id']} carries no evidence_spans")
                spans_tok = []
                if spans_char:
                    # single tokenization serves both ids and char->token maps
                    delta = text.index(raw) if args.chat_template else 0
                    enc = tok(text, add_special_tokens=False,
                              return_offsets_mapping=True)
                    ids = torch.tensor([enc["input_ids"]],
                                       dtype=torch.long, device="cuda")
                    spans_tok = spans_to_token_ranges(
                        enc["offset_mapping"],
                        [(s + delta, e + delta) for s, e in spans_char])
                else:
                    ids = tok(text, add_special_tokens=False,
                              return_tensors="pt").input_ids.to("cuda")
                fmbl = prep_row = None
                if args.force_buckets == "static" or args.bias_arm != "off":
                    prep_row = torch.load(
                        Path(args.prep_dir) / f"{Path(suite_path).stem}.pt",
                        map_location="cpu", weights_only=False)
                    assert prep_row["T"] == int(ids.shape[1]), \
                        ("prep/ids length mismatch", prep_row["T"],
                         int(ids.shape[1]))
                    sha = hashlib.sha256(
                        json.dumps(ids[0].tolist()).encode()).hexdigest()
                    assert sha == prep_row["ids_sha256"], "prep/ids sha mismatch"
                if args.force_buckets == "static":
                    cap_len = int(ids.shape[1]) + mnt + args.reserve_pad
                    fmbl = {}
                    for li_s, dd in prep_row["layers"].items():
                        fn = {int(l): v for l, v in dd["static"].items()}
                        fmbl[int(li_s)] = force_bitmasks(
                            fn, cap_len, geom["compression_rate"],
                            geom["top_k"], "cuda")
                fnodes = fmasks = None
                if args.force_buckets in ("evidence", "random"):
                    cap_len = int(ids.shape[1]) + mnt + args.reserve_pad
                    nl_cap = compute_num_layers(
                        cap_len, geom["compression_rate"],
                        geom["max_top_nodes"])
                    fnodes = force_nodes_per_level(
                        spans_tok, nl_cap, geom["compression_rate"],
                        int(ids.shape[1]), args.force_buckets,
                        f"{args.force_seed}:{r['sample_id']}")
                    fmasks = force_bitmasks(
                        fnodes, cap_len, geom["compression_rate"],
                        geom["top_k"], "cuda")
                if args.trace_selection and spans_tok:
                    trace = SelectionTrace(
                        spans_tok, c=geom["compression_rate"],
                        top_k=geom["top_k"],
                        theta=theta,
                        head_dim=model.model.layers[0].self_attn.head_dim,
                        device="cuda",
                        out_path=out / f"sel_trace_{r['sample_id']}.jsonl",
                        force_nodes=None if fnodes is None else {
                            li: torch.tensor(v, dtype=torch.long)
                            for li, v in fnodes.items() if v})
                t0 = time.time()
                if trace is not None:
                    sess = TracingSession(model,
                                          reserve=mnt + args.reserve_pad,
                                          offload_leaves=args.offload_leaves,
                                          trace=trace)
                else:
                    sess = RecSADecodeSession(
                        model, reserve=mnt + args.reserve_pad,
                        offload_leaves=args.offload_leaves)
                if fmasks is not None:
                    if args.force_scope == "decode":
                        wrap_decode_scope(sess, acfg, fmasks)
                    else:
                        acfg["force_masks"] = fmasks
                if fmbl is not None:
                    wrap_decode_scope_by_layer(sess, acfg, fmbl)
                if args.bias_arm != "off":
                    wrap_decode_bias(sess, prep_row)
                out_ids = sess.generate(ids, mnt, eos_token_ids=eos,
                                        use_cuda_graph=args.graph)
                wall = time.time() - t0
                tree_levels = sess.states[0].num_layers
                tree_seq = sess.states[0].seq_len
                # generate() feeds back every emitted token except the last
                assert tree_seq == int(ids.shape[1]) + len(out_ids) - 1, (
                    f"tree_seq {tree_seq} != prefill {int(ids.shape[1])} + "
                    f"gen {len(out_ids)} - 1 — session/driver desync")
                pred = tok.decode(out_ids, skip_special_tokens=True)
                golds = r.get("answers") or [r["answer"]]
                match = r["metadata"].get("match", "all")
                score = score_row(pred, golds, match)
                s_tok = score_row_tok(pred, golds, match)
                trace_fields = trace.summary() if trace is not None else {}
                fout.write(json.dumps(dict(
                    eval_len=eval_len, sample_id=r["sample_id"],
                    task_type=r["task_type"],
                    position_bucket=r.get("position_bucket", "na"),
                    score=score,
                    score_bnd=score_row_bnd(pred, golds, match),
                    score_tok=s_tok,
                    score_tok_any=score_row_tok_any(pred, golds, match),
                    score_tok_credited=score_row_tok_credited(
                        pred, golds, match, s_tok),
                    em=float(normalize_prediction(pred) ==
                             normalize_prediction(str(golds[0]))),
                    mnt_used=mnt,
                    **({} if fmasks is None else dict(
                        force_arm=args.force_buckets,
                        force_scope=args.force_scope,
                        forced_per_level={str(li): len(v)
                                          for li, v in fnodes.items()})),
                    **({} if fmbl is None else dict(
                        force_arm="static", force_scope="decode")),
                    **({} if args.bias_arm == "off"
                       else dict(bias_arm=args.bias_arm)),
                    n_golds=len(golds),
                    **trace_fields,
                    prefill_tokens=int(ids.shape[1]), gen_tokens=len(out_ids),
                    tree_levels=tree_levels, tree_seq=tree_seq,
                    peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                    wall_s=round(wall, 1),
                    **({} if geom["summaries"] == "softmax"
                       else dict(summaries=geom["summaries"])),
                    # a summary_norm rule tags its rows (absent = off)
                    **({} if geom.get("summary_norm") is None
                       else dict(summary_norm=geom["summary_norm"])),
                    **({} if mode == "internal" else dict(graft_mode=mode)),
                    **({} if args.rope_scaling_json is None
                       else dict(rope_scaling=args.rope_scaling_json)),
                    pred=pred[:300], pred_full=pred)) + "\n")
                fout.flush()
                n_done += 1
                print(f"[{n_done}/{len(rows)}] {r['task_type']} "
                      f"score={score:.2f} {wall:.0f}s", flush=True)
            except Exception as e:
                fout.write(json.dumps(dict(
                    eval_len=eval_len, sample_id=r["sample_id"],
                    task_type=r.get("task_type"),
                    error=repr(e)[:500])) + "\n")
                fout.flush()
                print(f"[ERROR] {r['sample_id']}: {repr(e)[:200]}", flush=True)
                if "CUDA error" in str(e):
                    raise
                torch.cuda.empty_cache()
            finally:
                acfg.pop("force_masks", None)
                acfg.pop("force_masks_by_layer", None)
                del sess

    rows_by_id, err_ids = {}, set()
    for l in open(per_sample):
        try:
            rr = json.loads(l)
        except json.JSONDecodeError:
            continue
        if "score" in rr:
            rows_by_id[rr["sample_id"]] = rr
        else:
            err_ids.add(rr.get("sample_id"))
    rows_out = list(rows_by_id.values())
    by_task = {}
    for r in rows_out:
        by_task.setdefault(r["task_type"], []).append(r)

    def agg(rs, k):
        vs = [r[k] for r in rs if k in r]
        return round(sum(vs) / len(vs), 4) if vs else None

    peaks = [r["peak_gib"] for r in rows_out if "peak_gib" in r]
    prov["prefill"] = prefill_config()
    summary = dict(
        model=args.model, graft=args.graft, probe_ckpt=args.probe_ckpt,
        top_k=geom["top_k"], q_chunk=args.q_chunk,
        summaries=geom["summaries"],
        summary_norm=geom.get("summary_norm"),
        graft_geometry=geom,
        rope_theta=theta_rec["rope_theta"],
        rope_theta_resolution=theta_rec["rope_theta_resolution"],
        rope_theta_expected=theta_rec["rope_theta_expected"],
        **({} if mode == "internal" else dict(graft_mode=mode)),
        **({} if args.rope_scaling_json is None else dict(rope_scaling=args.rope_scaling_json)),
        **({} if rotary_rec is None else dict(position_mode="global_true_position", rotary=rotary_rec,
                                              yarn_factor=rotary_rec["yarn_factor"])),
        build_chunk_parents=args.build_chunk_parents,
        mnt_default=args.mnt_default,
        instrument=dict(mnt_override=args.mnt_override or None,
                        extra_eos=args.extra_eos or None,
                        chat_template=bool(args.chat_template),
                        chat_template_kwargs=chat_kw or None,
                        chat_template_hybrid=(env_assert.hybrid_thinking_template(tok)
                                              if args.chat_template else None),
                        trace_selection=bool(args.trace_selection),
                        force_buckets=args.force_buckets,
                        force_scope=(args.force_scope
                                     if args.force_buckets != "off" else None),
                        force_seed=(args.force_seed
                                    if args.force_buckets == "random" else None),
                        bias_arm=(args.bias_arm
                                  if args.bias_arm != "off" else None),
                        prep_dir=(args.prep_dir or None)),
        decode=decode_label(args.offload_leaves, args.graph, geom["summaries"],
                            geom.get("summary_norm"), mode, args.rope_scaling_json),
        provenance=prov,
        suites=[dict(len=eval_len, path=suite_path, sha256=sha256(suite_path))],
        panel={str(eval_len): dict(
            peak_gib=round(max(peaks), 2) if peaks
            else round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            wall_s=round(time.time() - t_start, 1),
            errors_residual=sorted(str(i) for i in err_ids - set(rows_by_id)),
            all_task=dict(string_match=agg(rows_out, "score"),
                          string_match_bnd=agg(rows_out, "score_bnd"),
                          string_match_tok=agg(rows_out, "score_tok"),
                          string_match_tok_any=agg(rows_out, "score_tok_any"),
                          string_match_tok_credited=agg(
                              rows_out, "score_tok_credited"),
                          sel_fine_cov=agg(rows_out, "sel_fine_cov_mean"),
                          em=agg(rows_out, "em"), n=len(rows_out)),
            by_task={t: dict(string_match=agg(rs, "score"),
                             string_match_bnd=agg(rs, "score_bnd"),
                             string_match_tok=agg(rs, "score_tok"),
                             string_match_tok_any=agg(rs, "score_tok_any"),
                             string_match_tok_credited=agg(
                                 rs, "score_tok_credited"),
                             sel_fine_cov=agg(rs, "sel_fine_cov_mean"),
                             em=agg(rs, "em"), n=len(rs))
                     for t, rs in sorted(by_task.items())})})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"PEAK-MEM {summary['panel'][str(eval_len)]['peak_gib']} GiB")
    print("DECODE-RULER-DONE")


if __name__ == "__main__":
    main()
