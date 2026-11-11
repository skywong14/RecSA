#!/usr/bin/env python3
"""Dense FlashAttention decode latency (CUDA-graphed, batch 1) and a model-side-only decode step."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "benchmarks"))

import bench_decode_latency as bdl  # noqa: E402

DENSE_MODEL = os.environ.get("MODEL_DIR", "Qwen/Qwen3-4B")   # setup/get_model.sh: Qwen/Qwen3-4B @1cfa9a72
DENSE_YARN = {32768: None, 131072: (4.0, 32768), 262144: (8.0, 32768), 524288: (16.0, 32768)}
KV_BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2
N_PROFILE = bdl.N_PROFILE
SEED = bdl.SEED
GIB = float(1 << 30)


class ModelSide:
    """The model side of the RecSA decode step (recsa/decode.py) with a pluggable attention."""

    def __init__(self, model, fused_norm=None, fused_silu=None):
        import torch
        import recsa.decode as HD
        from recsa.decode import fused_qkv_weight
        self.fn = bool(HD.RECSA_DECODE_FUSED_NORM if fused_norm is None else fused_norm)
        self.fs = bool(HD.RECSA_DECODE_FUSED_SILU if fused_silu is None else fused_silu)
        self.m = model
        self.layers = list(model.model.layers)
        a0 = self.layers[0].self_attn
        self.hd = a0.head_dim
        self.nq = a0.q_proj.weight.shape[0]
        self.nk = a0.k_proj.weight.shape[0]
        self.W = [fused_qkv_weight(l.self_attn) for l in self.layers]
        if any(w is None for w in self.W):
            raise RuntimeError("fused q/k/v buffers missing: call recsa.decode.fuse_qkv_weights(model) first")
        self.torch = torch

    def layer(self, li, h, attend):
        """One decoder layer. attend(li, q, k, v) -> o [B, S, Hq, hd] or None (= model_only: o_proj consumes q)."""
        import torch.nn.functional as F
        from recsa.decode_fused_norm import mlp_forward, rms_norm
        layer = self.layers[li]
        attn = layer.self_attn
        B, S = h.shape[:2]
        fn = self.fn
        hn = rms_norm(layer.input_layernorm, h, fn)
        qkv = F.linear(hn, self.W[li])
        k = rms_norm(attn.k_norm, qkv[..., self.nq:self.nq + self.nk].view(B, S, -1, self.hd), fn)
        v = qkv[..., self.nq + self.nk:].view(B, S, -1, self.hd)
        q = rms_norm(attn.q_norm, qkv[..., :self.nq].view(B, S, -1, self.hd), fn)
        o = attend(li, q, k, v) if attend is not None else q
        h = h + attn.o_proj(o.reshape(B, S, -1))
        return h + mlp_forward(layer.mlp, rms_norm(layer.post_attention_layernorm, h, fn), self.fs)

    def logits(self, h):
        from recsa.decode_fused_norm import rms_norm
        mdl = self.m.model
        return self.m.lm_head(rms_norm(mdl.norm, h, self.fn))[:, -1].float()


class GraphedStep:
    """A whole decode step as one CUDA graph. body() reads self.tok and returns logits."""

    def __init__(self, body, tok):
        import torch
        self.torch, self.body, self.tok = torch, body, tok
        self.g, self.out = None, None
        self.a9_captured = None

    def capture(self, warm=2):
        torch = self.torch
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warm):
                self.body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        from recsa.decode_fused_norm import CALLS
        c0 = dict(CALLS)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.out = self.body()
        self.a9_captured = (CALLS["norm"] - c0["norm"], CALLS["silu"] - c0["silu"])
        self.g = g
        torch.cuda.synchronize()

    def step(self, token_id):
        self.tok.copy_(token_id)
        self.g.replay()
        return self.out.clone()


class DenseSession:
    """Static [1, cap, Hkv, hd] bf16 K/V cache per layer + flash_attn_with_kvcache."""

    def __init__(self, ms: ModelSide, cap: int, cos, sin):
        import torch
        self.ms, self.torch, self.cap = ms, torch, cap
        a0 = ms.layers[0].self_attn
        hkv = a0.k_proj.weight.shape[0] // ms.hd
        dt, dev = a0.q_proj.weight.dtype, a0.q_proj.weight.device
        self.kc = [torch.zeros(1, cap, hkv, ms.hd, dtype=dt, device=dev) for _ in ms.layers]
        self.vc = [torch.zeros(1, cap, hkv, ms.hd, dtype=dt, device=dev) for _ in ms.layers]
        self.seqlens = torch.zeros(1, dtype=torch.int32, device=dev)
        self.cos, self.sin = cos, sin
        self.scale = a0.scaling
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.t = 0

    def attend_fn(self, num_splits, causal=True):
        from flash_attn import flash_attn_with_kvcache

        def attend(li, q, k, v):
            return flash_attn_with_kvcache(q, self.kc[li], self.vc[li], k=k, v=v, rotary_cos=self.cos,
                                           rotary_sin=self.sin, cache_seqlens=self.seqlens,
                                           softmax_scale=self.scale, causal=causal, rotary_interleaved=False,
                                           num_splits=num_splits)
        return attend

    def forward_tokens(self, ids, num_splits=0, want_logits=True, ms=None):
        """Run len(ids) tokens through every layer (prefill chunk or eager step); advance the cursor."""
        ms = ms or self.ms
        h = ms.m.model.embed_tokens(ids)
        att = self.attend_fn(num_splits)
        for li in range(len(ms.layers)):
            h = ms.layer(li, h, att)
        out = ms.logits(h) if want_logits else None
        self.seqlens.add_(ids.shape[1])
        self.t += ids.shape[1]
        return out

    def prefill(self, ids, chunk):
        torch = self.torch
        T = ids.shape[1]
        out = None
        for c0 in range(0, T, chunk):
            out = self.forward_tokens(ids[:, c0:c0 + chunk], want_logits=c0 + chunk >= T)
        torch.cuda.synchronize()
        assert self.t == T and int(self.seqlens.item()) == T
        return out

    def body_fn(self, num_splits, ms=None):
        ms, torch = (ms or self.ms), self.torch

        def body():
            h = ms.m.model.embed_tokens(self.tok)
            att = self.attend_fn(num_splits)
            for li in range(len(ms.layers)):
                h = ms.layer(li, h, att)
            out = ms.logits(h)
            self.seqlens.add_(1)
            return out
        return body

    def attn_only_graph(self, num_splits, n_layers=None, layer_map=None):
        """CUDA graph of the per-layer flash_attn_with_kvcache calls alone, on fixed random q/k/v. cache_seqlens is not
        advanced, so every replay writes the same slot."""
        torch, ms = self.torch, self.ms
        from flash_attn import flash_attn_with_kvcache
        a0 = ms.layers[0].self_attn
        hq, hkv = self.ms.nq // ms.hd, self.kc[0].shape[2]
        g0 = torch.Generator(device="cuda")
        g0.manual_seed(7)
        q = torch.randn(1, 1, hq, ms.hd, device="cuda", generator=g0).to(self.kc[0].dtype)
        k = torch.randn(1, 1, hkv, ms.hd, device="cuda", generator=g0).to(self.kc[0].dtype)
        v = torch.randn(1, 1, hkv, ms.hd, device="cuda", generator=g0).to(self.kc[0].dtype)
        lm = layer_map or list(range(len(ms.layers)))

        def body():
            o = None
            for li in lm:
                o = flash_attn_with_kvcache(q, self.kc[li], self.vc[li], k=k, v=v, rotary_cos=self.cos,
                                            rotary_sin=self.sin, cache_seqlens=self.seqlens,
                                            softmax_scale=a0.scaling, causal=True, rotary_interleaved=False,
                                            num_splits=num_splits)
            return o
        gs = GraphedStep(body, torch.empty(1, device="cuda"))
        gs.capture()
        return gs


def time_replays(gs, n, warm=8):
    torch = gs.torch
    for _ in range(warm):
        gs.g.replay()
    torch.cuda.synchronize()
    per = []
    for _ in range(n):
        t0 = time.perf_counter()
        gs.g.replay()
        torch.cuda.synchronize()
        per.append((time.perf_counter() - t0) * 1e3)
    return per


def profile_steps(step_fn, n=N_PROFILE, name_groups=None):
    """GPU busy time, launches and top kernels over n steps. Run after all timing."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for j in range(n):
            step_fn(j)
        torch.cuda.synchronize()
    dev = bdl.device_events(prof)
    by = {}
    for s, e, nm in dev:
        c = by.setdefault(nm, [0, 0])
        c[0] += 1
        c[1] += e - s
    top = sorted(by.items(), key=lambda kv: -kv[1][1])[:15]
    rec = dict(gpu_busy_ms=round(sum(e - s for s, e, _ in dev) / 1e6 / n, 3),
               launches_per_step=round(len(dev) / n, 1),
               top_kernels=[dict(name=nm[:160], calls_per_step=round(c / n, 2), us_per_step=round(ns / 1e3 / n, 1))
                            for nm, (c, ns) in top])
    for gname, pred in (name_groups or {}).items():
        sel = [(nm, c, ns) for nm, (c, ns) in by.items() if pred(nm)]
        rec[gname] = dict(ms_per_step=round(sum(ns for _, _, ns in sel) / 1e6 / n, 4),
                          launches_per_step=round(sum(c for _, c, _ in sel) / n, 2),
                          kernels=sorted({nm[:120] for nm, _, _ in sel}))
    return rec


def is_flash(nm):
    return "flash" in nm.lower()


A9_GROUPS = {"a9_norm_kernels": lambda nm: "recsa_decode_rmsnorm" in nm,
             "a9_silu_kernels": lambda nm: "recsa_decode_silu_mul" in nm}


def a9_census(prof_fused, prof_unfused, n_layers, fused_tag, unfused_tag, cap_fused=None, cap_unfused=None):
    n_norm, n_silu = 4 * n_layers + 1, n_layers
    got = dict(fused_norm=prof_fused["a9_norm_kernels"]["launches_per_step"],
               fused_silu=prof_fused["a9_silu_kernels"]["launches_per_step"],
               unfused_norm=prof_unfused["a9_norm_kernels"]["launches_per_step"],
               unfused_silu=prof_unfused["a9_silu_kernels"]["launches_per_step"],
               launches_delta=round(prof_unfused["launches_per_step"] - prof_fused["launches_per_step"], 2))
    want = dict(fused_norm=n_norm, fused_silu=n_silu, unfused_norm=0, unfused_silu=0,
                launches_delta=7 * n_norm + n_silu)
    got["launches_delta_matches"] = float(got["launches_delta"]) == float(want["launches_delta"])
    got["launches_delta_judged"] = False
    ok = got["unfused_norm"] == 0 and got["unfused_silu"] == 0
    ok &= all(float(got[k]) <= want[k] for k in ("fused_norm", "fused_silu"))
    got["profiler_coverage_fused"] = round(min(float(got[k]) / want[k] for k in ("fused_norm", "fused_silu")), 4)
    cap = dict(fused=list(cap_fused) if cap_fused is not None else None,
               unfused=list(cap_unfused) if cap_unfused is not None else None)
    if cap_fused is not None:
        ok &= tuple(cap_fused) == (n_norm, n_silu)
    if cap_unfused is not None:
        ok &= tuple(cap_unfused) == (0, 0)
    return dict(arms=[fused_tag, unfused_tag], measured=got, expected=want, captured=cap,
                verdict="PASS" if ok else "MISMATCH")


def rope_table(model, cfg, T, cap, yarn):
    """cos/sin [cap, hd/2] (bf16) from HF's Qwen3RotaryEmbedding for this length's rope_parameters."""
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    if yarn is None:
        rot = model.model.rotary_emb
        rp = dict(getattr(model.config, "rope_parameters", None) or {})
    else:
        c2 = copy.deepcopy(cfg)
        rp = dict(getattr(c2, "rope_parameters", None) or {})
        if "rope_theta" not in rp and getattr(c2, "rope_theta", None):
            rp["rope_theta"] = c2.rope_theta
        rp.update({"rope_type": "yarn", "factor": yarn[0], "original_max_position_embeddings": yarn[1]})
        c2.rope_parameters = rp
        c2.max_position_embeddings = max(c2.max_position_embeddings, cap)
        rot = Qwen3RotaryEmbedding(c2, device="cuda")
    pos = torch.arange(cap, device="cuda")[None]
    x = torch.empty(1, dtype=torch.bfloat16, device="cuda")
    cos, sin = rot(x, pos)
    hd2 = cos.shape[-1] // 2
    assert torch.equal(cos[0, :, :hd2], cos[0, :, hd2:])
    rec = dict(rope_parameters={k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                                for k, v in rp.items()},
               attention_scaling=float(getattr(rot, "attention_scaling", 1.0)), yarn=yarn)
    return cos[0, :, :hd2].contiguous(), sin[0, :, :hd2].contiguous(), rec


def heuristic_splits(T_cap, num_sms, hkv=8, hd=128):
    """flash-attn 2.8 num_splits_heuristic for decode, re-implemented since the C++ value is not exposed."""
    block_n = 256 if hd <= 64 else (128 if hd <= 128 else 64)
    n_blocks = (T_cap + block_n - 1) // block_n
    bnm, sms = hkv * 1, num_sms * 2
    if bnm >= 0.8 * sms:
        return 1
    mx = min(128, sms, n_blocks)
    cd = lambda a, b: (a + b - 1) // b  # noqa: E731
    elig = lambda s: s == 1 or cd(n_blocks, s) != cd(n_blocks, s - 1)  # noqa: E731
    eff = []
    for s in range(1, mx + 1):
        if not elig(s):
            eff.append(0.0)
        else:
            w = bnm * s / sms
            eff.append(w / -(-w // 1))
    best = max(eff)
    for s in range(1, mx + 1):
        if elig(s) and eff[s - 1] >= 0.85 * best:
            return s
    return 1


def build_dense(path, cap):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from recsa.decode import fuse_qkv_weights
    cfg = AutoConfig.from_pretrained(path)
    cfg.max_position_embeddings = max(cfg.max_position_embeddings, cap)
    model = AutoModelForCausalLM.from_pretrained(path, config=cfg, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()
    n = fuse_qkv_weights(model)
    return model, cfg, n


def alt_timing(arms, steps, block, log=None):
    """arms: {name: (prep, run)}. prep stages the next token untimed; the timed region is run() + synchronize."""
    import torch
    order = list(arms)
    per = {a: [] for a in order}
    for bi in range(steps // block):
        rot = order[bi % len(order):] + order[:bi % len(order)]
        for a in rot:
            prep, run = arms[a]
            w0 = time.time()
            for _ in range(block):
                prep()
                t0 = time.perf_counter()
                run()
                torch.cuda.synchronize()
                per[a].append((time.perf_counter() - t0) * 1e3)
            if log is not None:
                log.append([a, round(w0, 4), round(time.time(), 4)])
    return per


def graph_arm(g, nxt):
    """(prep, run) for a GraphedStep."""
    import torch
    tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    return (lambda: tok.fill_(nxt())), (lambda: g.step(tok))


def run_dense_length(args, model, cfg, ms, mo, T, rec, ms_off=None, mo_off=None):
    import torch
    cap = T + args.reserve
    yarn = DENSE_YARN.get(T, None) if not args.no_yarn else None
    cos, sin, rrec = rope_table(model, cfg, T, cap, yarn)
    uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    rec.update(T=T, cap=cap, rope=rrec, gpu_tenants_start=bdl.gpu_tenants(uuid),
               kv_bytes=T * KV_BYTES_PER_TOKEN, kv_gib=round(T * KV_BYTES_PER_TOKEN / GIB, 2),
               flash_heuristic_splits=heuristic_splits(cap, nsm))
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(SEED)
    ids = torch.randint(1000, cfg.vocab_size - 1000, (1, T + args.reserve), device="cuda")
    stream = ids[0, T:].tolist()
    cur = [0]

    def nxt():
        v = stream[cur[0]]
        cur[0] += 1
        return v
    s = DenseSession(ms, cap, cos, sin)
    rec["cache_alloc_gib"] = round(torch.cuda.memory_allocated() / GIB, 2)
    t0 = time.perf_counter()
    with torch.inference_mode():
        if args.prefill == "real":
            s.prefill(ids[:, :T], args.prefill_chunk)
        else:
            for li in range(len(ms.layers)):
                s.kc[li][:, :T].normal_()
                s.vc[li][:, :T].normal_()
            s.seqlens.fill_(T)
            s.t = T
        torch.cuda.synchronize()
        rec["prefill"] = args.prefill
        rec["prefill_s"] = round(time.perf_counter() - t0, 1)

        # attention-only sweep (graph of 36 per-layer calls) -> tuned num_splits
        sweep = {}
        for ns in args.splits:
            try:
                ag = s.attn_only_graph(ns)
                sweep[ns] = bdl.stats(time_replays(ag, args.sweep_reps))
                del ag
            except Exception as e:  # noqa: BLE001
                sweep[ns] = dict(error=f"{type(e).__name__}: {e}")
        ok = {ns: v for ns, v in sweep.items() if "median" in v}
        best = min(ok, key=lambda ns: ok[ns]["median"])
        rec["attn_only_sweep_ms"] = {str(k): v for k, v in sweep.items()}
        rec["num_splits_best"] = best
        if args.tuned_splits >= 0:
            tuned = args.tuned_splits if args.tuned_splits != 0 else None
            rec["tuned_splits_source"] = (f"a priori (--tuned-splits {args.tuned_splits}); "
                                          f"this process's attention-only sweep best = {best}")
        else:
            tuned = best if best != 0 else None
            rec["tuned_splits_source"] = "this process's attention-only sweep (fastest median)"
        rec["tuned_splits"] = tuned

        # graphs over the shared cache/cursor; model_only on its own buffers
        graphs = {"dense": GraphedStep(s.body_fn(0), s.tok)}
        if tuned is not None and tuned != rec["flash_heuristic_splits"]:
            graphs["dense_tuned"] = GraphedStep(s.body_fn(tuned), s.tok)
        if ms_off is not None:
            graphs["dense_unfused"] = GraphedStep(s.body_fn(0, ms_off), s.tok)
            if "dense_tuned" in graphs:
                graphs["dense_tuned_unfused"] = GraphedStep(s.body_fn(tuned, ms_off), s.tok)
        for g in graphs.values():
            g.capture()
            s.t += 2
        tok = torch.empty(1, 1, dtype=torch.long, device="cuda")
        tok.fill_(nxt())
        s.tok.copy_(tok)
        le = s.forward_tokens(s.tok, 0).clone()
        s.seqlens.sub_(1)
        s.t -= 1
        lg = graphs["dense"].step(tok)
        s.t += 1
        rec["graph_equals_eager_one_step"] = bool(torch.equal(le, lg))
        if "dense_unfused" in graphs:
            pairs = [("dense", "dense_unfused")]
            if "dense_tuned_unfused" in graphs:
                pairs.append(("dense_tuned", "dense_tuned_unfused"))
            eq = []
            for _ in range(args.ab_check_steps):
                tok.fill_(nxt())
                for fa, fb in pairs:
                    la = graphs[fa].step(tok)
                    s.seqlens.sub_(1)
                    lb = graphs[fb].step(tok)
                    s.seqlens.sub_(1)
                    eq.append(bool(torch.equal(la, lb)))
                s.seqlens.add_(1)
                s.t += 1
            rec["side_check_fused_vs_unfused"] = dict(
                logits_equal_per_step=eq, verdict="BITWISE" if eq and all(eq) else "MISMATCH", pairs=pairs,
                protocol="per step and pair: fused graph, rewind the cursor, unfused graph on the same token, "
                         "rewind; torch.equal; then advance the cursor once")
        arms = {name: graph_arm(g, nxt) for name, g in graphs.items()}
        arms["model_only"] = graph_arm(mo, nxt)
        if mo_off is not None:
            arms["model_only_unfused"] = graph_arm(mo_off, nxt)
        for _ in range(args.warm):
            for a, (prep, run) in arms.items():
                prep()
                run()
        torch.cuda.synchronize()
        rec["window_log"] = dict(blocks=[], profiles=[])
        per = alt_timing(arms, args.steps, args.block, rec["window_log"]["blocks"])
        s.t = int(s.seqlens.item())
        rec["timing"] = {a: bdl.stats(p) for a, p in per.items()}
        rec["per_step_ms"] = {a: [round(x, 4) for x in p] for a, p in per.items()}
        rec["timing_protocol"] = (f"one process; arms {list(arms)} on one shared cache/cursor (model_only on its own "
                                  f"buffers), alternating blocks of {args.block} steps with rotating order, "
                                  f"{args.steps} timed steps per arm after {args.warm} warm per arm, sync per step; "
                                  "profiles after all timing")
        if args.b2b_steps > 0:
            b2b = {}
            for a, (prep, run) in arms.items():
                for _ in range(8):
                    prep()
                    run()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.b2b_steps):
                    prep()
                    run()
                torch.cuda.synchronize()
                b2b[a] = round((time.perf_counter() - t0) * 1e3 / args.b2b_steps, 4)
            rec["timing_back_to_back_ms"] = b2b
            rec["timing_back_to_back_protocol"] = (f"after the main timing, per arm 8 warm + {args.b2b_steps} "
                                                   "steps back to back (stage + replay + clone), one sync at the "
                                                   "end; mean")
        # profiles (after all timing)
        prof = {}
        for a, (prep, run) in arms.items():
            def sf(j, prep=prep, run=run):
                prep()
                run()
            w0 = time.time()
            prof[a] = profile_steps(sf, name_groups=dict(A9_GROUPS, flash_kernels=is_flash))
            rec["window_log"]["profiles"].append([a, round(w0, 4), round(time.time(), 4)])
        rec["profile"] = prof
        if "dense_unfused" in prof:
            nl = len(ms.layers)
            rec["census"] = dict(dense=a9_census(prof["dense"], prof["dense_unfused"], nl, "dense", "dense_unfused",
                                                 graphs["dense"].a9_captured, graphs["dense_unfused"].a9_captured),
                                 model_only=a9_census(prof["model_only"], prof["model_only_unfused"], nl,
                                                      "model_only", "model_only_unfused", mo.a9_captured,
                                                      mo_off.a9_captured))
            if "dense_tuned_unfused" in prof:
                rec["census"]["dense_tuned"] = a9_census(
                    prof["dense_tuned"], prof["dense_tuned_unfused"], nl, "dense_tuned", "dense_tuned_unfused",
                    graphs["dense_tuned"].a9_captured, graphs["dense_tuned_unfused"].a9_captured)
            rec["census"]["verdict"] = ("PASS" if all(v["verdict"] == "PASS" for k, v in rec["census"].items()
                                                      if isinstance(v, dict)) else "MISMATCH")
        ag = s.attn_only_graph(0)
        rec["attn_only_graph_heuristic_ms"] = bdl.stats(time_replays(ag, args.steps))
        del ag
        if tuned is not None:
            ag = s.attn_only_graph(tuned)
            rec["attn_only_graph_tuned_ms"] = bdl.stats(time_replays(ag, args.steps))
            del ag
    mom = rec["timing"]["model_only"]["median"]
    der = {}
    for a in graphs:
        d = rec["timing"][a]["median"] - mom
        fk = prof[a]["flash_kernels"]["ms_per_step"]
        der[a] = dict(attn_ms_differential=round(d, 3),
                      attn_ms_flash_kernels=round(fk, 3),
                      gpu_busy_minus_model_only=round(prof[a]["gpu_busy_ms"] - prof["model_only"]["gpu_busy_ms"], 3),
                      kv_bw_tbps_differential=round(T * KV_BYTES_PER_TOKEN / (d * 1e-3) / 1e12, 3),
                      kv_bw_tbps_flash_kernels=round(T * KV_BYTES_PER_TOKEN / (fk * 1e-3) / 1e12, 3))
    ao = rec["attn_only_graph_heuristic_ms"]["median"]
    der["attn_only_graph_heuristic"] = dict(ms=ao, kv_bw_tbps=round(T * KV_BYTES_PER_TOKEN / (ao * 1e-3) / 1e12, 3))
    if "attn_only_graph_tuned_ms" in rec:
        at = rec["attn_only_graph_tuned_ms"]["median"]
        der["attn_only_graph_tuned"] = dict(ms=at, kv_bw_tbps=round(T * KV_BYTES_PER_TOKEN / (at * 1e-3) / 1e12, 3))
    rec["derived"] = der
    rec["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 2)
    rec["gpu_tenants_end"] = bdl.gpu_tenants(uuid)
    graphs = None
    s = None
    gc.collect()
    torch.cuda.empty_cache()


def synthetic_attn(args, ms, T, rec):
    """Attention only, on one layer's synthetic [1, T+64, 8, 128] cache, called 36 times in one graph."""
    import torch
    cap = T + 64
    hd2 = ms.hd // 2
    pos = torch.arange(cap, device="cuda", dtype=torch.float32)[:, None]
    inv = 1.0 / (1e6 ** (torch.arange(0, ms.hd, 2, device="cuda", dtype=torch.float32) / ms.hd))
    cos, sin = (pos * inv).cos().to(torch.bfloat16), (pos * inv).sin().to(torch.bfloat16)
    s = DenseSession.__new__(DenseSession)
    s.ms, s.torch, s.cap = ms, torch, cap
    s.kc = [torch.randn(1, cap, 8, ms.hd, dtype=torch.bfloat16, device="cuda")]
    s.vc = [torch.randn(1, cap, 8, ms.hd, dtype=torch.bfloat16, device="cuda")]
    s.seqlens = torch.full((1,), T, dtype=torch.int32, device="cuda")
    s.cos, s.sin = cos[:, :hd2].contiguous(), sin[:, :hd2].contiguous()
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    out = dict(T=T, label="SYNTHETIC: one layer's random cache, 36 calls per graph replay",
               flash_heuristic_splits=heuristic_splits(cap, nsm), sweep_ms={})
    with torch.inference_mode():
        for ns in args.splits:
            ag = s.attn_only_graph(ns, layer_map=[0] * len(ms.layers))
            st = bdl.stats(time_replays(ag, args.sweep_reps))
            st["kv_bw_tbps"] = round(T * KV_BYTES_PER_TOKEN / (st["median"] * 1e-3) / 1e12, 3)
            out["sweep_ms"][str(ns)] = st
            del ag
    rec.append(out)
    s = None
    gc.collect()
    torch.cuda.empty_cache()


def run_recsa_length(args, model, cfg, mo, T, rec):
    import torch
    n_after = args.warm + args.steps + N_PROFILE + 8
    reserve = n_after + 16
    torch.manual_seed(SEED)
    ids = torch.randint(1000, cfg.vocab_size - 1000, (1, T + n_after + 8), device="cuda")
    uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    rec.update(T=T, gpu_tenants_start=bdl.gpu_tenants(uuid))
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    a = bdl.Arm("default", model, ids, T, reserve, False, graph=True)
    rec["recsa_session"] = a.rec
    j = [0]

    def h_prep():
        a.tok.fill_(a.stream[j[0]])
        j[0] += 1
    cnt = iter(range(10 ** 9))
    arms = {"recsa": (h_prep, lambda: a.s.step(a.tok)), "model_only": graph_arm(mo, lambda: 1000 + next(cnt) % 1000)}
    with torch.inference_mode(), bdl.arm_scope("default"):
        for _ in range(args.warm):
            for prep, run in arms.values():
                prep()
                run()
        torch.cuda.synchronize()
        rec["recsa_session"]["paths"] = a.paths()
        per = alt_timing(arms, args.steps, args.block)
        rec["timing"] = {k: bdl.stats(v) for k, v in per.items()}
        rec["per_step_ms"] = {k: [round(x, 4) for x in v] for k, v in per.items()}
        rec["timing_protocol"] = (f"one process; recsa default session (bench_decode_latency Arm) and model_only on "
                                  f"the same model object, alternating blocks of {args.block}, {args.steps} timed "
                                  f"steps each after {args.warm} warm, sync per step; profiles after all timing")
        prof = {}
        for name, (prep, run) in arms.items():
            def sf(jj, prep=prep, run=run):
                prep()
                run()
            prof[name] = profile_steps(sf, name_groups=(
                {"recsa_named_kernels": lambda nm: "recsa" in nm.lower()} if name == "recsa" else None))
    rec["profile"] = prof
    d = rec["timing"]["recsa"]["median"] - rec["timing"]["model_only"]["median"]
    rec["derived"] = dict(attn_ms_differential=round(d, 3),
                          gpu_busy_minus_model_only=round(prof["recsa"]["gpu_busy_ms"]
                                                          - prof["model_only"]["gpu_busy_ms"], 3))
    rec["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 2)
    rec["gpu_tenants_end"] = bdl.gpu_tenants(uuid)
    a = None
    gc.collect()
    torch.cuda.empty_cache()


def make_model_only(ms):
    import torch
    tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    cnt = torch.zeros(1, dtype=torch.int32, device="cuda")

    def body():
        h = ms.m.model.embed_tokens(tok)
        for li in range(len(ms.layers)):
            h = ms.layer(li, h, None)
        out = ms.logits(h)
        cnt.add_(1)       # the cursor advance the decode steps also do
        return out
    g = GraphedStep(body, tok)
    with torch.inference_mode():
        g.capture()
    return g


def run_check(args, rec):
    """Real-text prompt at args.check_len: dense prefill + graphed greedy decode vs HF eager decode (sdpa and
    flash_attention_2), teacher-forced on the dense tokens."""
    import torch
    from transformers import AutoTokenizer, DynamicCache
    T = args.check_len
    cap = T + 256
    model, cfg, nf = build_dense(args.model, cap)
    ms = ModelSide(model)
    tokz = AutoTokenizer.from_pretrained(args.model)
    text = "".join(Path(p).read_text() for p in args.check_text)
    ids_all = tokz(text, return_tensors="pt").input_ids.to("cuda")
    assert ids_all.shape[1] >= T, f"check text has {ids_all.shape[1]} tokens < {T}"
    ids = ids_all[:, :T]
    rec.update(T=T, text_files=args.check_text, text_tokens=int(ids_all.shape[1]), fused_layers=nf,
               prompt_head=tokz.decode(ids[0, :40]))
    cos, sin, rrec = rope_table(model, cfg, T, cap, None)
    rec["rope"] = rrec
    N = args.check_steps
    with torch.inference_mode():
        s = DenseSession(ms, cap, cos, sin)
        p_d = s.prefill(ids, args.prefill_chunk)
        g = GraphedStep(s.body_fn(0), s.tok)
        first = int(p_d.argmax(-1).item())
        s.tok.fill_(first)
        g.capture()
        s.seqlens.fill_(T)
        s.t = T
        toks, Ld = [first], []
        tok = torch.empty(1, 1, dtype=torch.long, device="cuda")
        for i in range(N):
            tok.fill_(toks[-1])
            lg = g.step(tok)
            Ld.append(lg)
            toks.append(int(lg.argmax(-1).item()))
        # graph == eager on the same state
        tok.fill_(toks[-1])
        s.tok.copy_(tok)
        le = s.forward_tokens(s.tok, 0).clone()
        s.seqlens.sub_(1)
        lg2 = g.step(tok)
        rec["graph_equals_eager_one_step"] = bool(torch.equal(le, lg2))
        s = g = None
        gc.collect()
        torch.cuda.empty_cache()
        refs = {}
        for impl in ("sdpa", "flash_attention_2"):
            try:
                if model.config._attn_implementation != impl:
                    model.set_attn_implementation(impl)
                try:
                    cache = DynamicCache(config=model.config)
                except TypeError:
                    cache = DynamicCache()
                out = model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
                P = out.logits[:, -1].float()
                L = []
                for i in range(N):
                    out = model(torch.tensor([[toks[i]]], device="cuda"), past_key_values=cache, use_cache=True)
                    L.append(out.logits[:, -1].float())
                refs[impl] = (P, L)
                cache = out = None
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                rec[f"ref_{impl}_error"] = f"{type(e).__name__}: {e}"
        if model.config._attn_implementation != "sdpa":
            model.set_attn_implementation("sdpa")

    def cmp(a, b):
        d = (a - b).abs()
        la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        kl = float((la.exp() * (la - lb)).sum())
        t5a = set(a.topk(5, -1).indices[0].tolist())
        t5b = set(b.topk(5, -1).indices[0].tolist())
        return dict(argmax_equal=bool(a.argmax(-1).item() == b.argmax(-1).item()), max_abs=round(float(d.max()), 4),
                    mean_abs=round(float(d.mean()), 5), kl=float(f"{kl:.3e}"), top5_overlap=len(t5a & t5b))
    res = {}
    if "sdpa" in refs:
        P, L = refs["sdpa"]
        res["dense_vs_sdpa"] = dict(prefill=cmp(p_d, P), steps=[cmp(x, y) for x, y in zip(Ld, L)])
    if "sdpa" in refs and "flash_attention_2" in refs:
        P, L = refs["sdpa"]
        P2, L2 = refs["flash_attention_2"]
        res["hf_fa2_vs_sdpa"] = dict(prefill=cmp(P2, P), steps=[cmp(x, y) for x, y in zip(L2, L)])
    for k, v in res.items():
        st = v["steps"]
        v["summary"] = dict(argmax_agree=f"{sum(x['argmax_equal'] for x in st)}/{len(st)}",
                            max_abs=max(x["max_abs"] for x in st), max_kl=max(x["kl"] for x in st),
                            min_top5_overlap=min(x["top5_overlap"] for x in st))
    rec["compare"] = res
    rec["dense_greedy_tokens"] = toks
    rec["dense_greedy_text"] = tokz.decode(toks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("dense", "recsa"), default="dense")
    ap.add_argument("--model", default=DENSE_MODEL, help="dense HF model (default: original Qwen3-4B snapshot)")
    ap.add_argument("--export", default=None, help="--arm recsa: RecSA export dir")
    ap.add_argument("--lengths", default="", help="comma-separated context lengths")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--warm", type=int, default=32)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--reserve", type=int, default=1024, help="dense cache slots past T (decode steps)")
    ap.add_argument("--splits", default="0,8,16,24,32,48,64,96,128", help="num_splits sweep (0 = flash heuristic)")
    ap.add_argument("--sweep-reps", type=int, default=32)
    ap.add_argument("--tuned-splits", type=int, default=-1,
                    help="fix the tuned arm's num_splits a priori (>= 0; 0 = heuristic only); default -1 = this "
                         "process's sweep best")
    ap.add_argument("--b2b-steps", type=int, default=0,
                    help="dense: extra back-to-back reading of N steps per arm after the main timing")
    ap.add_argument("--prefill", choices=("real", "random"), default="real")
    ap.add_argument("--prefill-chunk", type=int, default=16384)
    ap.add_argument("--no-yarn", action="store_true")
    ap.add_argument("--synthetic-attn-lengths", default="", help="attention-only synthetic one-layer microbench")
    ap.add_argument("--check", action="store_true", help="correctness sanity instead of timing")
    ap.add_argument("--check-len", type=int, default=32768)
    ap.add_argument("--check-steps", type=int, default=16)
    ap.add_argument("--check-text", default=str(REPO / "recsa" / "parallel.py"),
                    help="comma-separated text files of the --check prompt (>= --check-len tokens)")
    ap.add_argument("--q-chunk", type=int, default=131072)
    ap.add_argument("--model-side-ab", action="store_true",
                    help="dense: also time the unfused model side (dense_unfused, model_only_unfused), side check "
                         "and launch census")
    ap.add_argument("--ab-check-steps", type=int, default=24,
                    help="--model-side-ab: steps whose logits are compared fused vs unfused")
    ap.add_argument("--out", required=True)
    ap.add_argument("--node-note", default="", help="free text: what else ran on the node")
    args = ap.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "recsa":
        sys.exit("bench_decode_latency_dense: run inside the `recsa` conda env; "
                 f"CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV')!r}")
    args.splits = [int(x) for x in args.splits.split(",") if x.strip()]
    args.check_text = [p for p in args.check_text.split(",") if p]
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    syn = [int(x) for x in args.synthetic_attn_lengths.split(",") if x.strip()]
    if args.steps % args.block:
        ap.error("--steps must be divisible by --block")

    import torch
    import triton
    import transformers
    import flash_attn
    out = Path(args.out)
    res = dict(schema="recsa_dense_decode_latency/1", argv=sys.argv, arm=args.arm,
               env=dict(bdl.env_record(torch, triton, transformers), flash_attn=flash_attn.__version__),
               dtype="bfloat16", batch=1, node_note=args.node_note,
               start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), runs=[], synthetic=[])

    def dump():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    envp = out.with_name(out.stem + "_env.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    envp.write_text(json.dumps(res["env"], indent=1, default=str) + "\n")
    rc = 0
    if args.check:
        rec = {}
        res["check"] = rec
        try:
            run_check(args, rec)
        except Exception as e:  # noqa: BLE001
            rec.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            rc = 1
        res["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        dump()
        print(json.dumps({k: v.get("summary") for k, v in rec.get("compare", {}).items()}),
              rec.get("graph_equals_eager_one_step"), rec.get("error", ""), flush=True)
        return rc

    cap = max(lengths + [0]) + max(args.reserve, 256)
    t0 = time.perf_counter()
    if args.arm == "dense":
        model, cfg, nf = build_dense(args.model, cap)
        res.update(model=args.model, fused_qkv_layers=nf)
    else:
        probe = str(Path(args.export) / "probe.pt") if (Path(args.export) / "probe.pt").exists() else None
        model, cfg, geom, theta_rec = bdl.build_model(args.export, cap, args.q_chunk, probe)
        from recsa.decode import fuse_qkv_weights
        nf = fuse_qkv_weights(model)
        res.update(export=str(Path(args.export).resolve()), probe_ckpt=probe, geometry=geom, rope_theta=theta_rec,
                   fused_qkv_layers=nf)
    ms = ModelSide(model)
    mo = make_model_only(ms)
    ms_off = mo_off = None
    if args.model_side_ab and args.arm == "dense":
        ms_off = ModelSide(model, fused_norm=False, fused_silu=False)
        mo_off = make_model_only(ms_off)
    res["model_build_s"] = round(time.perf_counter() - t0, 1)
    res["model_side"] = ("one input_layernorm per layer, fused q/k/v GEMM (recsa.decode."
                         "fuse_qkv_weights incl. canary); RMSNorm and silu*mul through "
                         "recsa.decode_fused_norm's helpers under the RecSA session's switches (both arms); "
                         "residuals / projections / lm_head = HF modules")
    res["model_side_switches"] = dict(fused_norm=ms.fn, fused_silu=ms.fs,
                                      unfused_ab_arm=ms_off is not None)
    dump()
    for T in syn:
        try:
            synthetic_attn(args, ms, T, res["synthetic"])
        except Exception as e:  # noqa: BLE001
            res["synthetic"].append(dict(T=T, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()))
            rc = 1
        dump()
    for T in lengths:
        rec = {}
        res["runs"].append(rec)
        try:
            if args.arm == "dense":
                run_dense_length(args, model, cfg, ms, mo, T, rec, ms_off, mo_off)
                if "MISMATCH" in (rec.get("census", {}).get("verdict"),
                                  rec.get("side_check_fused_vs_unfused", {}).get("verdict")):
                    rc = 1
            else:
                run_recsa_length(args, model, cfg, mo, T, rec)
        except Exception as e:  # noqa: BLE001
            rec.update(T=T, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            rc = 1
            gc.collect()
            torch.cuda.empty_cache()
        med = {a: v.get("median") for a, v in rec.get("timing", {}).items()}
        print(f"[dense_bench] arm={args.arm} T={T} median ms/token {med} derived={rec.get('derived')} "
              f"{rec.get('error', '')}", flush=True)
        dump()
    res["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump()
    return rc


if __name__ == "__main__":
    sys.exit(main())
