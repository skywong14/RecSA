"""Dense (FlashAttention-2) prefill wall time at 1M / 2M tokens on one H200."""
import argparse
import gc
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
OOM_RC = 42
STOP_RC = 43


def env_record():
    import torch
    import triton
    rec = {
        "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
        "sys.executable": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
        "cpu_affinity": f"{min(os.sched_getaffinity(0))}-{max(os.sched_getaffinity(0))} ({len(os.sched_getaffinity(0))} cpus)",
        "git_rev": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "git_dirty_code": len(subprocess.run(["git", "status", "--porcelain", "--", "recsa", "scripts"],
                                             capture_output=True, text=True).stdout.splitlines()),
        "cwd": os.getcwd(),
    }
    try:
        import transformers
        rec["transformers"] = transformers.__version__
    except Exception as e:  # pragma: no cover
        rec["transformers"] = f"unavailable: {e}"
    try:
        import flash_attn
        rec["flash_attn"] = flash_attn.__version__
    except Exception as e:  # pragma: no cover
        rec["flash_attn"] = f"unavailable: {e}"
    return rec


CHUNK_STATS = {"chunked_layer_calls": 0, "attention_calls": 0, "max_attn_T": 0}


def _chunked_decoder_layer_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                                   use_cache=False, position_embeddings=None, **kwargs):
    """Same math as transformers 5.9 Qwen3DecoderLayer.forward + Qwen3Attention.forward, token-wise ops in chunks."""
    import torch
    from transformers.models.qwen3 import modeling_qwen3 as mq
    if attention_mask is not None or past_key_values is not None or use_cache:
        raise RuntimeError("chunked layer supports only the no-padding, no-cache prefill (mask None, use_cache False)")
    B, T, _ = hidden_states.shape
    if B != 1:
        raise RuntimeError("batch 1 only")
    attn = self.self_attn
    hd = attn.head_dim
    nh = attn.config.num_attention_heads
    nkv = attn.config.num_key_value_heads
    C = int(self._seq_chunk)
    cos, sin = position_embeddings
    dev, dt = hidden_states.device, hidden_states.dtype
    q = torch.empty(B, T, nh, hd, device=dev, dtype=dt)
    k = torch.empty(B, T, nkv, hd, device=dev, dtype=dt)
    v = torch.empty(B, T, nkv, hd, device=dev, dtype=dt)
    for s in range(0, T, C):
        e = min(T, s + C)
        x = self.input_layernorm(hidden_states[:, s:e])
        shp = (B, e - s, -1, hd)
        qc = attn.q_norm(attn.q_proj(x).view(shp)).transpose(1, 2)
        kc = attn.k_norm(attn.k_proj(x).view(shp)).transpose(1, 2)
        vc = attn.v_proj(x).view(shp)
        qc, kc = mq.apply_rotary_pos_emb(qc, kc, cos[:, s:e], sin[:, s:e])
        q[:, s:e].copy_(qc.transpose(1, 2))
        k[:, s:e].copy_(kc.transpose(1, 2))
        v[:, s:e].copy_(vc)
        del x, qc, kc, vc
    attention_interface = mq.ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation,
                                                                     mq.eager_attention_forward)
    o, _ = attention_interface(attn, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attention_mask,
                               dropout=0.0, scaling=attn.scaling, sliding_window=attn.sliding_window,
                               position_ids=position_ids, use_cache=use_cache, **kwargs)
    CHUNK_STATS["attention_calls"] += 1
    CHUNK_STATS["max_attn_T"] = max(CHUNK_STATS["max_attn_T"], T)
    del q, k, v
    for s in range(0, T, C):
        e = min(T, s + C)
        a = attn.o_proj(o[:, s:e].reshape(B, e - s, -1))
        h = hidden_states[:, s:e] + a
        del a
        m = self.mlp(self.post_attention_layernorm(h))
        hidden_states[:, s:e] = h + m
        del h, m
    del o
    CHUNK_STATS["chunked_layer_calls"] += 1
    return hidden_states


def install_seq_chunking(model, chunk):
    n = 0
    for layer in model.model.layers:
        layer._seq_chunk = int(chunk)
        layer.forward = types.MethodType(_chunked_decoder_layer_forward, layer)
        n += 1
    return n


def remove_seq_chunking(model):
    for layer in model.model.layers:
        if "forward" in layer.__dict__:
            del layer.forward


def load_dense(model_path, max_len):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_path)
    cfg.max_position_embeddings = max(cfg.max_position_embeddings, max_len + 512)
    return AutoModelForCausalLM.from_pretrained(model_path, config=cfg, torch_dtype=torch.bfloat16,
                                                attn_implementation="flash_attention_2").cuda().eval()


def make_ids(toks, L):
    import numpy as np
    import torch
    E = len(toks)
    return torch.from_numpy(np.asarray(toks[E - L:E], dtype=np.int64)).view(1, L).cuda()


def dense_forward(dmodel, ids):
    import torch
    with torch.inference_mode():
        o = dmodel(ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
    return o


def timed(dmodel, ids):
    """One timed dense forward, as in prefill_bench.py."""
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    o = dense_forward(dmodel, ids)
    dt = time.perf_counter() - t0
    del o
    return dt, torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens-npy", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--mode", choices=["plain", "chunked", "auto"], default="auto")
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--warmup-len", type=int, default=32768)
    ap.add_argument("--stop-file", default=str(HERE / "STOP"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import torch
    toks = np.load(a.tokens_npy, mmap_mode="r")
    E = len(toks)
    lengths = [int(x) for x in a.lengths.split(",")]
    assert max(lengths + [a.warmup_len]) <= E, (max(lengths), E)
    rec = {"schema": "dense_prefill_bench_v1", "argv": sys.argv, "env": env_record(), "tokens_npy": a.tokens_npy,
           "tokens_len": E, "model": a.model, "mode_requested": a.mode, "chunk": a.chunk,
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "instrument": "dmodel(ids, use_cache=False, logits_to_keep=1) under inference_mode + cuda.synchronize; "
                         "input = last L tokens; peak = max_memory_allocated after reset", "lengths": {}}
    out = Path(a.out)

    def save():
        rec["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out.write_text(json.dumps(rec, indent=1))

    t0 = time.perf_counter()
    dmodel = load_dense(a.model, max(lengths))
    rec["load_s"] = time.perf_counter() - t0
    rec["weights_gib"] = torch.cuda.memory_allocated() / 2**30
    rec["attn_implementation"] = dmodel.config._attn_implementation
    wids = make_ids(toks, a.warmup_len)

    def warm(tag):
        ws = []
        for _ in range(a.warmup):
            t = time.perf_counter()
            o = dense_forward(dmodel, wids)
            del o
            ws.append(time.perf_counter() - t)
        rec.setdefault("warmup", []).append({"tag": tag, "len": a.warmup_len, "times_s": ws})
        torch.cuda.empty_cache()

    mode = a.mode
    if mode == "chunked":
        rec["chunked_layers"] = install_seq_chunking(dmodel, a.chunk)
    warm(mode)
    for L in lengths:
        if os.path.exists(a.stop_file):
            rec["stopped"] = f"STOP file before L={L}"
            save()
            sys.exit(STOP_RC)
        ids = make_ids(toks, L)
        row = {"L": L}
        times, peaks, peaks_res = [], [], []
        used = "chunked" if mode == "chunked" else "plain"
        for r in range(a.reps):
            c0 = dict(CHUNK_STATS)
            oom = None
            try:
                dt, pk, pr = timed(dmodel, ids)
            except torch.OutOfMemoryError as ex:
                if used != "plain":
                    raise
                oom = str(ex).splitlines()[0][:400]
            if oom is not None:
                gc.collect()
                torch.cuda.synchronize()
                row["plain_oom"] = {"error": oom, "peak_alloc_gib_at_oom": torch.cuda.max_memory_allocated() / 2**30}
                print(f"L={L}: plain forward OOM: {oom}", flush=True)
                torch.cuda.empty_cache()
                if mode == "plain":
                    rec["lengths"][str(L)] = row
                    save()
                    sys.exit(OOM_RC)
                # auto: switch to chunked, repeat the warm-up on the chunked path, then time it.
                rec["chunked_layers"] = install_seq_chunking(dmodel, a.chunk)
                used = "chunked"
                warm("chunked_after_plain_oom")
                c0 = dict(CHUNK_STATS)
                dt, pk, pr = timed(dmodel, ids)
            times.append(dt)
            peaks.append(pk)
            peaks_res.append(pr)
            row.setdefault("chunk_stats_delta", []).append(
                {k: CHUNK_STATS[k] - c0[k] for k in ("chunked_layer_calls", "attention_calls")})
            print(f"L={L} rep {r}: {used} {dt:.3f} s, peak alloc {pk:.2f} GiB", flush=True)
        row.update({"mode_used": used, "chunk": a.chunk if used == "chunked" else None, "times_s": times,
                    "median_s": statistics.median(times), "min_s": min(times), "peak_alloc_gib": max(peaks),
                    "peak_reserved_gib": max(peaks_res), "median_us_per_token": 1e6 * statistics.median(times) / L,
                    "max_attn_T_seen": CHUNK_STATS["max_attn_T"] if used == "chunked" else None})
        rec["lengths"][str(L)] = row
        save()
        del ids
        torch.cuda.empty_cache()
    save()


if __name__ == "__main__":
    main()
