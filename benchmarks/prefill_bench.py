"""Prefill wall time: the RecSA session prefill against the dense flash_attention_2 forward (end-to-end model prefill, one
GPU)."""
import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def env_record():
    import torch
    import triton
    return {
        "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
        "sys.executable": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "RECSA_PREFILL_FAST": os.environ.get("RECSA_PREFILL_FAST", "(unset = default on)"),
        "git_rev": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "git_dirty_code": len(subprocess.run(["git", "status", "--porcelain", "--", "recsa", "scripts"],
                                             capture_output=True, text=True).stdout.splitlines()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens-npy", default=os.path.join(os.environ.get("RECSA_DATA", "third_party/data"),
                                                    "pg19_tokens/tokens/pg19_00000_537_E1000000.npy"))
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--q-chunk", type=int, default=131072)
    ap.add_argument("--dense-max", type=int, default=524288)
    ap.add_argument("--offload-above", type=int, default=524288)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from graft import install_recsa_attention, install_streamed_forward
    import env_assert
    import graft_geometry
    from recsa.decode import RecSADecodeSession
    toks = np.load(a.tokens_npy, mmap_mode="r")
    E = len(toks)
    lengths = [int(x) for x in a.lengths.split(",")]
    assert max(lengths) <= E, (max(lengths), E)
    rec = {"schema": "prefill_bench_v1", "argv": sys.argv, "env": env_record(), "tokens_npy": a.tokens_npy,
           "tokens_len": E, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "lengths": {}}
    geom = graft_geometry.resolve(a.model, top_k=None)
    rec["geometry"] = {k: geom[k] for k in ("compression_rate", "top_k", "max_top_nodes")}
    cfg = AutoConfig.from_pretrained(a.model)
    cfg.max_position_embeddings = max(cfg.max_position_embeddings, max(lengths) + 512)
    need_dense = any(L <= a.dense_max for L in lengths)
    dmodel = None
    if need_dense:
        dmodel = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, torch_dtype=torch.bfloat16,
                                                      attn_implementation="flash_attention_2").cuda().eval()
    model = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    mode, theta, _ = env_assert.graft_rope_base("internal:auto", model.config)
    rec["rope"] = {"mode": mode, "theta": theta}
    install_recsa_attention(model, mode=mode, rope_base=theta, probe_ckpt=str(Path(a.model) / "probe.pt"),
                            compression_rate=geom["compression_rate"], top_k=geom["top_k"],
                            max_top_nodes=geom["max_top_nodes"], q_chunk=a.q_chunk)
    install_streamed_forward(model, q_chunk=a.q_chunk)
    for L in lengths:
        ids = torch.from_numpy(np.asarray(toks[E - L:E], dtype=np.int64)).view(1, L).cuda()
        off = L > a.offload_above
        run_dense = dmodel is not None and L <= a.dense_max

        def dense():
            with torch.inference_mode():
                o = dmodel(ids, use_cache=False, logits_to_keep=1)
                torch.cuda.synchronize()
            del o

        def recsa():
            with torch.inference_mode():
                sess = RecSADecodeSession(model, reserve=64, offload_leaves=off)
                lg = sess.prefill(ids)
                torch.cuda.synchronize()
            del sess, lg

        arms = [("dense", dense)] if run_dense else []
        arms.append(("recsa", recsa))
        for _ in range(a.warmup):
            for _, f in arms:
                f()
        times = {n: [] for n, _ in arms}
        peak = {}
        for _ in range(a.reps):
            for n, f in arms:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                f()
                times[n].append(time.perf_counter() - t0)
                peak[n] = max(peak.get(n, 0.0), torch.cuda.max_memory_allocated() / 2**30)
        row = {"offload_leaves": off, "dense_run": run_dense}
        for n in times:
            row[n] = {"times_s": times[n], "median_s": statistics.median(times[n]), "min_s": min(times[n]),
                      "peak_alloc_gib": peak[n], "median_ms_per_token": 1e3 * statistics.median(times[n]) / L}
        if run_dense:
            row["recsa_over_dense"] = row["recsa"]["median_s"] / row["dense"]["median_s"]
        rec["lengths"][str(L)] = row
        msg = f"L={L}: recsa med {row['recsa']['median_s']:.3f} s (offload {off})"
        if run_dense:
            msg += f"  dense med {row['dense']['median_s']:.3f} s  recsa/dense {row['recsa_over_dense']:.3f}"
        print(msg, flush=True)
        rec["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        Path(a.out).write_text(json.dumps(rec, indent=1))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
