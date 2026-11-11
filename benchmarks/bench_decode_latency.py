#!/usr/bin/env python3
"""Per-token decode latency of a RecSA export (CUDA-graphed, batch 1): the default decode path and the path with every
decode optimisation switch at its fallback value."""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

SEED = 20260731          # the decode profilers' prompt seed
N_PROFILE = 8
GIB = float(1 << 30)

SWITCHES = (
    ("recsa.parallel", "RECSA_DSPLIT_FINALIZE", "RECSA_DSPLIT_FINALIZE", "v3", "legacy"),
    ("recsa.decode", "RECSA_DECODE_SINGLE_NORM", "RECSA_DECODE_SINGLE_NORM", True, False),
    ("recsa.decode", "RECSA_DECODE_FUSED_QKV", "RECSA_DECODE_FUSED_QKV", True, False),
    ("recsa.probe_pool", "RECSA_MLP_CURSOR_POOL", "RECSA_MLP_CURSOR_POOL", "v2", "legacy"),
    ("recsa.probe_pool", "RECSA_MLP_POOL_SEGMENTS", "RECSA_MLP_POOL_SEGMENTS", "fused", "torch"),
    ("recsa.parallel", "RECSA_DSPLIT_CHAIN", "RECSA_DSPLIT_CHAIN", "fused", "legacy"),
    ("recsa.parallel", "RECSA_DSPLIT_HFUSE", "RECSA_DSPLIT_HFUSE", "on", "off"),
    ("recsa.incremental_tree", "MLP_OFFLOAD_GRAPH", "RECSA_MLP_OFFLOAD_GRAPH", "on", "off"),
    ("recsa.decode", "RECSA_DECODE_FUSED_NORM", "RECSA_DECODE_FUSED_NORM", True, False),
    ("recsa.decode", "RECSA_DECODE_FUSED_SILU", "RECSA_DECODE_FUSED_SILU", True, False),
    ("recsa.dsplit_score_v2", "RECSA_DSPLIT_SCORE", "RECSA_DSPLIT_SCORE", "v2", "v1"),
    ("recsa.dsplit_score_v2", "RECSA_DSPLIT_SCORE_ROLE", "RECSA_DSPLIT_SCORE_ROLE", "noscat", "v1"),
    ("recsa.mlp_pool_fused_v2", "RECSA_MLP_POOL_FUSED2", "RECSA_MLP_POOL_FUSED2", "on", "off"),
    ("recsa.incremental_tree", "OFFLOAD_POOL_V2", "RECSA_OFFLOAD_POOL_V2", "off", "off"),
    ("recsa.decode", "RECSA_OFFLOAD_MIRROR", "RECSA_OFFLOAD_MIRROR", "off", "off"),
)
ARM_VALUES = {"default": {s[2]: s[3] for s in SWITCHES}, "legacy": {s[2]: s[4] for s in SWITCHES}}
LEGACY_FLIPS = None
FLAT2_ARM = "flat2"
FLAT2_SHARED_DSPLIT = ("recsa_dsplit_bottom_part_kernel", "recsa_dsplit_merge_pair_kernel")


def switch_values(arm: str) -> dict:
    return ARM_VALUES["default" if arm in ("default", FLAT2_ARM) else "legacy"]


def set_legacy_flips(names):
    global LEGACY_FLIPS
    known = {s[2]: s for s in SWITCHES}
    if names is not None:
        vals = {}
        for item in names:
            nm, eq, val = (x.strip() for x in item.partition("="))
            if nm not in known:
                raise ValueError(f"unknown switch {nm!r}; known: {sorted(known)}")
            if not eq:
                vals[nm] = known[nm][4]
            elif isinstance(known[nm][3], bool):
                if val.lower() not in ("0", "1", "true", "false"):
                    raise ValueError(f"{nm}={val!r}: a boolean switch takes 0/1/true/false")
                vals[nm] = val.lower() in ("1", "true")
            else:
                vals[nm] = val.lower()
        LEGACY_FLIPS = vals
    else:
        LEGACY_FLIPS = None
    ARM_VALUES["legacy"] = {s[2]: (s[4] if LEGACY_FLIPS is None else LEGACY_FLIPS.get(s[2], ARM_VALUES["default"][s[2]]))
                            for s in SWITCHES}


def set_default_overrides(items):
    known = {s[2]: s for s in SWITCHES}
    for item in items:
        nm, eq, val = (x.strip() for x in item.partition("="))
        if nm not in known or not eq:
            raise ValueError(f"--default-switches takes NAME=VALUE of known switches, got {item!r}")
        if isinstance(known[nm][3], bool):
            if val.lower() not in ("0", "1", "true", "false"):
                raise ValueError(f"{nm}={val!r}: a boolean switch takes 0/1/true/false")
            ARM_VALUES["default"][nm] = val.lower() in ("1", "true")
        else:
            ARM_VALUES["default"][nm] = val.lower()


@contextlib.contextmanager
def arm_scope(arm: str):
    """Set every switch's module global to the arm's value, then restore the imported values."""
    saved = []
    try:
        for mod, name, env, dflt, leg in SWITCHES:
            m = importlib.import_module(mod)
            saved.append((m, name, getattr(m, name)))
            setattr(m, name, switch_values(arm)[env])
        yield
    finally:
        for m, name, v in reversed(saved):
            setattr(m, name, v)


def sha16_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def git(*a) -> bytes:
    try:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return b""


def env_record(torch, triton, transformers) -> dict:
    """Environment fields plus the git state of the tree."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "recsa", "tests", "benchmarks").decode().split()
    return dict(
        host=socket.gethostname(), CONDA_DEFAULT_ENV=os.environ.get("CONDA_DEFAULT_ENV"),
        sys_executable=sys.executable, python=platform.python_version(), torch=torch.__version__,
        triton=triton.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
        gpu_name=props.name, gpu_uuid=str(getattr(props, "uuid", "")),
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
        TRITON_CACHE_DIR=os.environ.get("TRITON_CACHE_DIR"),
        repo=str(REPO), git_rev=git("rev-parse", "HEAD").decode().strip(),
        git_dirty=bool(git("status", "--porcelain", "--untracked-files=no").strip()),
        decode_diff_sha16=sha16_bytes(git("diff", "--", "recsa", "tests", "scripts/graft.py")),
        untracked_sha16={f: sha16_bytes((REPO / f).read_bytes()) for f in untracked},
        recsa_env={k: v for k, v in os.environ.items() if k.startswith("RECSA_")},
    )


def gpu_tenants(uuid: str) -> dict:
    """Memory used and compute apps on this GPU, from nvidia-smi."""
    def q(*a):
        try:
            return subprocess.run(["nvidia-smi", *a, "--format=csv,noheader,nounits"], capture_output=True,
                                  text=True, timeout=30).stdout.splitlines()
        except Exception as e:  # noqa: BLE001
            return [f"unavailable ({type(e).__name__})"]
    u = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    mem = [l for l in q("--query-gpu=uuid,memory.used") if l.startswith(u)]
    apps = [l for l in q("--query-compute-apps=gpu_uuid,pid,used_memory") if l.startswith(u)]
    return dict(memory_used_mib=mem, compute_apps=apps)


def build_model(export: str, cap: int, q_chunk: int, probe_ckpt, graft: str = "internal:auto", rs=None):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    import env_assert
    import graft_geometry
    from graft import install_recsa_attention, install_streamed_forward
    geom = graft_geometry.resolve(export, top_k=None)   # the export's trained geometry
    cfg = AutoConfig.from_pretrained(export)
    if rs is not None:
        import rope_scaling
        rope_scaling.merge_rope_scaling(cfg, rs)
    cfg.max_position_embeddings = max(cfg.max_position_embeddings, cap)
    model = AutoModelForCausalLM.from_pretrained(export, config=cfg, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()
    mode, theta, theta_rec = env_assert.graft_rope_base(graft, model.config)
    if mode == "external":
        import rope_scaling
        theta_rec = dict(theta_rec, position_mode="global_true_position",
                         rotary=rope_scaling.check_rotary(model, rs, theta=theta))
    install_recsa_attention(model, mode=mode, rope_base=theta, probe_ckpt=probe_ckpt,
                            compression_rate=geom["compression_rate"], top_k=geom["top_k"],
                            max_top_nodes=geom["max_top_nodes"], q_chunk=q_chunk)
    install_streamed_forward(model, q_chunk=q_chunk)
    return model, cfg, geom, theta_rec


def stats(per):
    s = sorted(per)
    return dict(n=len(s), median=round(statistics.median(s), 3), p90=round(s[int(0.9 * len(s))], 3),
                min=round(s[0], 3), mean=round(statistics.mean(s), 3))


def device_events(prof):
    """[(start_ns, end_ns, name)] of every device kernel / memcpy / memset in a finished torch.profiler run."""
    from torch.autograd import DeviceType
    out = []
    kr = getattr(getattr(prof, "profiler", None), "kineto_results", None)
    if kr is not None:
        for e in kr.events():
            try:
                if e.device_type() != DeviceType.CUDA or e.name().startswith("ProfilerStep"):
                    continue
                s = int(e.start_ns())
                out.append((s, s + int(e.duration_ns()), e.name()))
            except Exception:  # noqa: BLE001
                continue
    if not out:
        for e in prof.events():
            if e.device_type == DeviceType.CUDA and not getattr(e, "is_user_annotation", False):
                out.append((int(e.time_range.start * 1000), int(e.time_range.end * 1000), e.name))
    return out


class Arm:
    """One decode session of one arm, with its token stream and records."""

    def __init__(self, arm, model, ids, T, reserve, offload, graph):
        import torch
        self.arm, self.T, self.torch = arm, T, torch
        self.stream = ids[0, T:].tolist()
        self.tok = torch.empty(1, 1, dtype=torch.long, device="cuda")
        self.rec = dict(arm=arm, offload=offload)
        from recsa.decode import RecSADecodeSession
        with arm_scope(arm):
            self.s = RecSADecodeSession(model, reserve=reserve, offload_leaves=offload)
            t0 = time.perf_counter()
            self.s.prefill(ids[:, :T])
            torch.cuda.synchronize()
            self.rec["prefill_s"] = round(time.perf_counter() - t0, 1)
            self.rec["graph"] = False
            if graph:
                try:
                    self.s.enable_cuda_graph()
                    self.rec["graph"] = True
                except ValueError as e:
                    if not (arm == "legacy" and offload):
                        raise
                    self.rec["graph_refused"] = f"{e} -> the legacy offload MLP arm runs eager"
        st = self.s.states
        self.rec.update(nl=st[0].num_layers, n_layers=len(st))

    def steps(self, j0, n, timed=False, keep=False):
        torch = self.torch
        per, lg = [], []
        with arm_scope(self.arm), torch.inference_mode():
            for j in range(j0, j0 + n):
                self.tok.fill_(self.stream[j])
                t0 = time.perf_counter()
                out = self.s.step(self.tok)
                if timed:
                    torch.cuda.synchronize()
                    per.append((time.perf_counter() - t0) * 1e3)
                if keep:
                    lg.append(out.float().cpu())
        torch.cuda.synchronize()
        return per, lg

    def paths(self):
        """Which decode paths the session ran."""
        st = self.s.states
        return dict(graph_captured=getattr(self.s, "_graph", None) is not None,
                    a2_layers_inline_fused=list(self.s.a2_last_step),
                    a5_defer_v_layers=sum(bool(getattr(x, "_defer_v", False)) for x in st),
                    a8_exact_layers=sum(bool(getattr(x, "_exact", False)) for x in st),
                    pool_fused_layers=sum(bool(getattr(x, "_fused", False)) for x in st),
                    a7_staged_layers=sum(getattr(x, "_stage_k", None) is not None for x in st),
                    offload_v2_layers=sum(bool(getattr(x, "_offload_v2", False)) for x in st),
                    mirror_layers=sum(getattr(x, "_mirror", None) is not None for x in st),
                    mirror_info=getattr(self.s, "mirror_info", None),
                    a9_fused_norm_silu_calls=list(getattr(self.s, "a9_last_step", (0, 0))))

    def profile(self, j0):
        torch = self.torch
        from torch.profiler import ProfilerActivity, profile
        with arm_scope(self.arm), torch.inference_mode():
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for j in range(j0, j0 + N_PROFILE):
                    self.tok.fill_(self.stream[j])
                    self.s.step(self.tok)
                torch.cuda.synchronize()
        dev = device_events(prof)
        by = {}
        for s, e, nm in dev:
            c = by.setdefault(nm, [0, 0])
            c[0] += 1
            c[1] += e - s
        top = sorted(by.items(), key=lambda kv: -kv[1][1])[:15]
        a9 = {g: dict(launches_per_step=round(sum(c for nm, (c, _) in by.items() if key in nm) / N_PROFILE, 2),
                      us_per_step=round(sum(ns for nm, (_, ns) in by.items() if key in nm) / 1e3 / N_PROFILE, 1))
              for g, key in (("a9_norm_kernels", "recsa_decode_rmsnorm"), ("a9_silu_kernels", "recsa_decode_silu_mul"))}
        return dict(gpu_busy_ms=round(sum(e - s for s, e, _ in dev) / 1e6 / N_PROFILE, 3),
                    launches_per_step=round(len(dev) / N_PROFILE, 1),
                    top_kernels=[dict(name=nm[:160], calls_per_step=round(c / N_PROFILE, 2),
                                      us_per_step=round(ns / 1e3 / N_PROFILE, 1)) for nm, (c, ns) in top], **a9)


def _flat2():
    """Import and install benchmarks/flat2_decode.py on first use."""
    if str(REPO / "benchmarks") not in sys.path:
        sys.path.insert(0, str(REPO / "benchmarks"))
    import flat2_decode
    flat2_decode.install()
    return flat2_decode


class Flat2Arm(Arm):
    """--arms flat2: the default arm with the forced two-level attention in its decode steps."""

    def __init__(self, arm, model, ids, T, reserve, offload, graph):
        import torch
        F = self.F = _flat2()
        F.reset()                       # a previous length's caches and workspaces
        F.FLAT2.on, F.FLAT2.mode = False, "fast"
        a0 = model.model.layers[0].self_attn
        c, n_layers = int(a0._recsa_cfg["c"]), len(model.model.layers)
        cap1 = (T + reserve + c - 1) // c + 1          # IncrementalTreeState.preallocate, level 1
        gib = F.reserve_kcache(n_layers, cap1, a0.k_proj.out_features // a0.head_dim, a0.head_dim,
                               torch.device("cuda"))
        self._builds0, self._finite = F.FLAT2.kcache_builds, []
        super().__init__(arm, model, ids, T, reserve, offload, graph)
        self.rec["flat2"] = dict(mode=F.FLAT2.mode, sub=F.FLAT2.sub, refresh_tiles=F.FLAT2.refresh_tiles,
                                 summary_slices=F.summary_slices(cap1), top_k=a0._recsa_cfg["top_k"],
                                 kcache_reserved_gib=round(gib, 3), level1_capacity_reserved=cap1,
                                 level1_capacity_nodes=int(self.s.states[0]._k[1].shape[1]))

    def steps(self, j0, n, timed=False, keep=False):
        self.F.FLAT2.on = True
        try:
            per, lg = super().steps(j0, n, timed=timed, keep=keep)
        finally:
            self.F.FLAT2.on = False
        self._finite += [bool(self.torch.isfinite(x).all()) for x in lg]
        return per, lg

    def profile(self, j0):
        self.F.FLAT2.on = True
        try:
            prof = super().profile(j0)
            self.rec["flat2_check"] = self.check(j0 + N_PROFILE)
        finally:
            self.F.FLAT2.on = False
        return prof

    def check(self, j0):
        """Profile N_PROFILE more steps to check that the graph replays run the forced step."""
        torch, F = self.torch, self.F
        from torch.profiler import ProfilerActivity, profile
        with arm_scope(self.arm), torch.inference_mode():
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for j in range(j0, j0 + N_PROFILE):
                    self.tok.fill_(self.stream[j])
                    self.s.step(self.tok)
                torch.cuda.synchronize()
        names = [nm for _, _, nm in device_events(prof)]
        n_layers = self.rec["n_layers"]
        score = sum("flat2_score_cached_kernel" in nm for nm in names) / N_PROFILE
        deployed = sorted({nm[:80] for nm in names
                           if "recsa_dsplit_" in nm and not any(k in nm for k in FLAT2_SHARED_DSPLIT)})
        ovf = int(F.FLAT2.last["overflow"].max().item()) if F.FLAT2.last is not None else None
        rec = dict(kcache_builds=F.FLAT2.kcache_builds - self._builds0,
                   reserved_buffers_unused=len(F.FLAT2.kcache_pool), flat2_score_launches_per_step=score, deployed_only_dsplit_kernels=deployed,
                   candidate_overflow_last_layer=ovf, check_logits_finite=bool(self._finite) and all(self._finite),
                   nl_start=self.rec["nl"], nl_end=self.s.states[0].num_layers)
        ok = (rec["kcache_builds"] == n_layers and rec["reserved_buffers_unused"] == 0
              and n_layers - 1 <= score <= n_layers and not deployed and ovf == 0 and rec["check_logits_finite"]
              and rec["nl_start"] == rec["nl_end"]
              and self.rec["flat2"]["level1_capacity_nodes"] == self.rec["flat2"]["level1_capacity_reserved"])
        rec["verdict"] = "PASS" if ok else "FAIL"
        return rec


def run_length(args, model, cfg, T, offload, out_rec):
    import torch
    n_layers = len(model.model.layers)
    arms = args.arms
    ab = args.ab
    if ab == "auto":
        ab = "alt" if (offload or T <= args.alt_max_tokens) else "seq"
    if len(arms) == 1:
        ab = "single"
    order = ["default", "legacy"] if ab == "alt" else (arms if ab == "single" else args.seq_order)
    n_after = args.warm + args.steps + N_PROFILE + 8
    reserve = n_after + 16
    torch.manual_seed(SEED)
    ids = torch.randint(1000, cfg.vocab_size - 1000, (1, T + n_after + 8), device="cuda")
    uuid = str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "uuid", ""))
    out_rec.update(T=T, offload_leaves=offload, ab=ab, order=order, gpu_tenants_start=gpu_tenants(uuid), sessions={})
    torch.cuda.reset_peak_memory_stats()
    K, j_t, j_p = args.check_steps, args.warm, args.warm + args.steps
    logits = {}

    def finish(a, profile_it):
        if profile_it:
            prof = a.rec["profile"] = a.profile(j_p)
            prof["wall_minus_busy_ms"] = round(a.rec["timing"]["median"] - prof["gpu_busy_ms"], 3)
        out_rec["sessions"][a.arm] = a.rec

    if ab == "alt":
        sess = {}
        for arm in order:
            sess[arm] = Arm(arm, model, ids, T, reserve, offload, graph=True)
        for i in range(K):
            for arm in order:
                _, lg = sess[arm].steps(i, 1, keep=True)
                logits.setdefault(arm, []).extend(lg)
        for arm in order:
            sess[arm].steps(K, j_t - K)
            sess[arm].rec["paths"] = sess[arm].paths()
        per = {arm: [] for arm in order}
        j = j_t
        for bi in range(args.steps // args.block):
            for arm in (order if bi % 2 == 0 else order[::-1]):
                p, _ = sess[arm].steps(j, args.block, timed=True)
                per[arm] += p
            j += args.block
        for arm in order:
            sess[arm].rec["timing"] = stats(per[arm])
            sess[arm].rec["per_step_ms"] = [round(x, 4) for x in per[arm]]
        for arm in order:   # profiles after all timing
            finish(sess[arm], True)
        protocol = (f"same process, both sessions resident, alternating blocks of {args.block} steps (rotating "
                    f"order), {args.steps} timed steps each after {args.warm} warm, sync per step; profiles after "
                    "all timing")
        sess = None
    else:
        for idx, arm in enumerate(order):
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            a = (Flat2Arm if arm == FLAT2_ARM else Arm)(arm, model, ids, T, reserve, offload, graph=True)
            _, logits[arm] = a.steps(0, K, keep=True)
            a.steps(K, j_t - K)
            a.rec["paths"] = a.paths()
            per, _ = a.steps(j_t, args.steps, timed=True)
            a.rec["timing"] = stats(per)
            a.rec["per_step_ms"] = [round(x, 4) for x in per]
            a.rec["timed_before_any_profiler_window"] = idx == 0 or not args.profile_each
            finish(a, idx == len(order) - 1 or args.profile_each)
            a = None
        protocol = (f"sessions one after the other in one process, order {order}; each timed ({args.steps} steps "
                    f"after {args.warm} warm, sync per step) before any profiler window"
                    + ("" if not args.profile_each else "; DIAGNOSTIC --profile-each: only the first session's "
                       "timing precedes every profiler window"))
    gc.collect()
    torch.cuda.empty_cache()
    out_rec["timing_protocol"] = protocol
    out_rec["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 2)
    out_rec["gpu_tenants_end"] = gpu_tenants(uuid)
    if "default" in logits and "legacy" in logits:
        eq = [bool(torch.equal(x, y)) for x, y in zip(logits["default"], logits["legacy"])]
        d = out_rec["sessions"]["default"]["paths"]
        a2_ok = d["a2_layers_inline_fused"] == [n_layers, n_layers]
        out_rec["side_check"] = dict(logits_equal_per_step=eq, a2_engaged_all_layers=a2_ok,
                                     verdict="BITWISE" if eq and all(eq) else "MISMATCH")
        dt, lt = out_rec["sessions"]["default"]["timing"], out_rec["sessions"]["legacy"]["timing"]
        out_rec["delta_median_ms_default_minus_legacy"] = round(dt["median"] - lt["median"], 3)
        out_rec["speedup_legacy_over_default"] = round(lt["median"] / dt["median"], 3)
    out_rec["census"] = a9_census(out_rec["sessions"], n_layers, offload)


def a9_census(sessions, n_layers, offload):
    norm_env, silu_env = "RECSA_DECODE_FUSED_NORM", "RECSA_DECODE_FUSED_SILU"
    n_norm, n_silu = 4 * n_layers + 1, n_layers
    rec, ok = {}, True
    for arm, srec in sessions.items():
        vals = switch_values(arm)
        inline = vals["RECSA_DECODE_SINGLE_NORM"]
        want = dict(norm=n_norm if vals[norm_env] else 0, silu=n_silu if (vals[silu_env] and inline) else 0)
        calls = srec.get("paths", {}).get("a9_fused_norm_silu_calls")
        good = calls is not None and tuple(calls) == (want["norm"], want["silu"])
        prof = srec.get("profile")
        got = None
        if prof is not None:
            got = dict(norm=prof["a9_norm_kernels"]["launches_per_step"],
                       silu=prof["a9_silu_kernels"]["launches_per_step"])
            good &= all(float(got[k]) <= want[k] for k in want)
        rec[arm] = dict(measured_profiler=got, expected=want, python_calls_captured_step=calls, ok=good)
        ok &= good
    diff = {e for e in ARM_VALUES["default"] if ARM_VALUES["default"][e] != ARM_VALUES["legacy"][e]}
    if (diff and diff <= {norm_env, silu_env} and "profile" in sessions.get("default", {})
            and "profile" in sessions.get("legacy", {}) and not offload):
        want = (7 * n_norm if norm_env in diff else 0) + (n_silu if silu_env in diff else 0)
        got = round(sessions["legacy"]["profile"]["launches_per_step"]
                    - sessions["default"]["profile"]["launches_per_step"], 2)
        rec["launch_delta_legacy_minus_default"] = dict(
            measured=got, expected=want, matches=float(got) == float(want), judged=False,
            note=f"coverage only: torch.profiler drops 0-2 leading events per window (+-{2 / N_PROFILE} per step)")
    rec["verdict"] = "PASS" if ok else "MISMATCH"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="export dir (config.json, weights, recsa_graft.json, probe.pt)")
    ap.add_argument("--probe-ckpt", default=None, help="default: <export>/probe.pt when present")
    ap.add_argument("--lengths", required=True, help="comma-separated prompt lengths, e.g. 32768,131072,1000000")
    ap.add_argument("--arms", default="default,legacy",
                    help="default,legacy | default | legacy | flat2 (forced two-level step, alone)")
    ap.add_argument("--ab", choices=("auto", "alt", "seq"), default="auto")
    ap.add_argument("--alt-max-tokens", type=int, default=262144,
                    help="auto: resident lengths up to this run both sessions side by side")
    ap.add_argument("--seq-order", default="default,legacy")
    ap.add_argument("--offload-min-tokens", type=int, default=1_000_000,
                    help="lengths >= this run with leaves in pinned host RAM (--offload-leaves session)")
    ap.add_argument("--offload-leaves", action="store_true", help="offload at every length")
    ap.add_argument("--steps", type=int, default=64, help="timed steps per session")
    ap.add_argument("--warm", type=int, default=32, help="steps before timing (incl. the check steps)")
    ap.add_argument("--check-steps", type=int, default=24, help="steps whose logits are compared across arms")
    ap.add_argument("--block", type=int, default=16, help="alt mode: steps per alternating block")
    ap.add_argument("--profile-each", action="store_true", help="seq mode, DIAGNOSTIC: profile every session")
    ap.add_argument("--q-chunk", type=int, default=131072)
    ap.add_argument("--ab-switches", default="",
                    help="comma-separated switch env names the legacy arm flips (default: all of them)")
    ap.add_argument("--default-switches", default="",
                    help="NAME=VALUE[,...]: values the default arm runs instead of the shipped defaults")
    ap.add_argument("--graft", default="internal:auto",
                    help="internal:auto (default, the frontier-local RoPE graft) | external (the "
                         "true-position RoPE graft)")
    ap.add_argument("--rope-scaling-json", default=None,
                    help="--graft external only: eval-time rope scaling, e.g. "
                         '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.graft.partition(":")[0] not in ("internal", "external"):
        ap.error("--graft takes internal[:theta] or external")
    if args.rope_scaling_json is not None and args.graft.partition(":")[0] != "external":
        ap.error("--rope-scaling-json needs --graft external")
    if os.environ.get("CONDA_DEFAULT_ENV") != "recsa":
        sys.exit("bench_decode_latency: run inside the `recsa` conda env (conda activate recsa); "
                 f"CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV')!r}")
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    args.seq_order = [a.strip() for a in args.seq_order.split(",")]
    if not set(args.arms) <= {"default", "legacy", FLAT2_ARM} or not args.arms:
        ap.error("--arms takes default and/or legacy, or flat2 alone")
    if FLAT2_ARM in args.arms and len(args.arms) != 1:
        ap.error("--arms flat2 runs alone (one session per process; compare it with an --arms default run)")
    if len(args.arms) == 2 and sorted(args.seq_order) != ["default", "legacy"]:
        ap.error("--seq-order must name default and legacy")
    if not (0 < args.check_steps < args.warm) or args.steps % args.block:
        ap.error("need 0 < --check-steps < --warm and --steps divisible by --block")
    lengths = [int(x) for x in args.lengths.split(",")]
    flips = [x.strip() for x in args.ab_switches.split(",") if x.strip()]
    try:
        set_default_overrides([x.strip() for x in args.default_switches.split(",") if x.strip()])
        set_legacy_flips(flips or None)
    except ValueError as e:
        ap.error(str(e))
    probe = args.probe_ckpt or (str(Path(args.export) / "probe.pt") if (Path(args.export) / "probe.pt").exists()
                                else None)

    import torch
    import triton
    import transformers
    for mod in ("recsa.parallel", "recsa.decode", "recsa.probe_pool", "recsa.incremental_tree"):
        f = Path(importlib.import_module(mod).__file__).resolve()
        assert str(f).startswith(str(REPO)), f"{mod} imported from {f}, not {REPO}"
    if FLAT2_ARM in args.arms:
        f = Path(_flat2().__file__).resolve()
        assert str(f).startswith(str(REPO)), f"flat2_decode imported from {f}, not {REPO}"
    out = Path(args.out)
    res = dict(schema="recsa_decode_latency/1", argv=sys.argv, export=str(Path(args.export).resolve()),
               probe_ckpt=probe, env=env_record(torch, triton, transformers), arms=args.arms,
               switches={arm: switch_values(arm) for arm in ("default", "legacy", FLAT2_ARM) if arm in args.arms},
               prompt=f"torch.manual_seed({SEED}); randint(1000, vocab-1000); teacher-forced continuation",
               start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), runs=[])

    def dump():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str) + "\n")

    cap = max(lengths) + args.warm + args.steps + N_PROFILE + 8 + 256
    t0 = time.perf_counter()
    import rope_scaling
    rs = rope_scaling.parse_rope_scaling(args.rope_scaling_json)
    model, cfg, geom, theta_rec = build_model(args.export, cap, args.q_chunk, probe, graft=args.graft, rs=rs)
    res.update(model_build_s=round(time.perf_counter() - t0, 1), geometry=geom, rope_theta=theta_rec)
    dump()
    rc = 0
    for T in lengths:
        rec = {}
        res["runs"].append(rec)
        offload = args.offload_leaves or T >= args.offload_min_tokens
        try:
            run_length(args, model, cfg, T, offload, rec)
        except Exception as e:  # noqa: BLE001
            rec.update(T=T, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            rc = 1
            gc.collect()
            torch.cuda.empty_cache()
        line = {a: s.get("timing", {}).get("median") for a, s in rec.get("sessions", {}).items()}
        if "MISMATCH" in (rec.get("side_check", {}).get("verdict"), rec.get("census", {}).get("verdict")):
            rc = 1
        f2 = rec.get("sessions", {}).get(FLAT2_ARM)
        if f2 is not None and f2.get("flat2_check", {}).get("verdict") != "PASS":
            rc = 1
        print(f"[bench_decode_latency] T={T} offload={offload} ab={rec.get('ab')} median ms/token {line} "
              f"side_check={rec.get('side_check', {}).get('verdict')} census={rec.get('census', {}).get('verdict')} "
              + (f"flat2_check={f2.get('flat2_check', {}).get('verdict')} " if f2 is not None else "")
              + f"{rec.get('error', '')}", flush=True)
        dump()
    res["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump()
    return rc


if __name__ == "__main__":
    sys.exit(main())
