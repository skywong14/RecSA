"""Grafts the Triton RecSA attention onto a Hugging Face Qwen3 model (evaluation, trainable and streamed-offload paths)."""
from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from recsa.backward import (  # noqa: E402
    recsa_forward_probe_autograd,
    recsa_forward_triton_autograd,
)
from recsa.decode_fused_norm import rms_norm  # noqa: E402
from recsa.level_bias import compose_level_bias  # noqa: E402
from recsa.parallel import (  # noqa: E402
    canonical_summary_norm,
    recsa_build_pyramid,
    recsa_forward,
    summaries_off,
)
from recsa.probe_pool import MLP_MODES, build_tree_layers, unpack_probe  # noqa: E402
from recsa.true_rope import (apply_rope, check_graft_mode, positions_from,  # noqa: E402
                           rope_cos_sin)
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb  # noqa: E402


def _decode_q_pre(attn, ctx, slot, hidden_states):
    """The precomputed q projection in ctx[slot] (fused QKV decode path), or None; then q_proj runs here."""
    q_pre = ctx[slot] if len(ctx) > slot else None
    if q_pre is not None and (
            q_pre.shape[:-1] != hidden_states.shape[:-1]
            or q_pre.shape[-1] != attn.q_proj.out_features
            or q_pre.dtype != hidden_states.dtype
            or q_pre.device != hidden_states.device):
        raise ValueError(
            f"decode q_pre {tuple(q_pre.shape)} {q_pre.dtype} does not match "
            f"q_proj({tuple(hidden_states.shape)} {hidden_states.dtype})")
    return q_pre


def _decode_fused_q_norm(ctx, slot) -> bool:
    """True when ctx[slot] asks for q_norm through recsa.decode_fused_norm's fused RMSNorm."""
    return len(ctx) > slot and ctx[slot] is True


def _ext_q_cos_sin(attn, q, shared, positions):
    """External graft, decode / streamed paths: the (cos, sin) that rotate q."""
    if shared is not None:
        return shared
    return rope_cos_sin(attn._recsa_rotary, q, positions())


def _recsa_attn_forward(self, hidden_states, position_embeddings, attention_mask=None,
                        past_key_values=None, **kwargs):
    cfg = self._recsa_cfg
    B, T, _ = hidden_states.shape
    shape = (B, T, -1, self.head_dim)
    gctx = getattr(self, "_graph_ctx", None)
    if gctx is not None:
        st, gcfg = gctx[:2]
        if getattr(self, "_recsa_gate_b", None) is not None:
            raise NotImplementedError(
                "probe_gate_b is live but the CUDA-graph decode session has "
                "no b carrier")
        q_pre = _decode_q_pre(self, gctx, 2, hidden_states)
        if not _decode_fused_q_norm(gctx, 3):
            q = self.q_norm((self.q_proj(hidden_states) if q_pre is None
                             else q_pre).view(shape))
        else:
            q = rms_norm(self.q_norm, (self.q_proj(hidden_states) if q_pre is None
                                       else q_pre).view(shape), True)
        g_rope_mode = "internal"
        if gcfg["mode"] == "external":
            cos, sin = _ext_q_cos_sin(self, q, gctx[4] if len(gctx) > 4 else None,
                                      lambda: st._t_dev.view(1, 1))
            q = apply_rope(q, cos, sin)
            g_rope_mode = "none"
        with torch.cuda.device(q.device):
            out = recsa_forward(
                q.contiguous(), st._k[0], st._v[0],
                compression_rate=gcfg["c"], max_top_nodes=gcfg["mtn"],
                top_k_per_layer=gcfg["top_k"], scale=self.scaling,
                rope_base=float(gcfg["rope_base"]), rope_mode=g_rope_mode,
                layers_kv=(st._k, st._v), q_start=0,
                t_dev=st._t_dev, cursor_nl=st.num_layers,
                no_summaries=summaries_off(gcfg.get("summaries")),
                summary_norm=gcfg.get("summary_norm"),
            )
        return self.o_proj(out.reshape(B, T, -1)), None
    ctx = getattr(self, "_streamed_ctx", None)
    if ctx is not None:
        k_full, v_full, pyramid, q0 = ctx[:4]
        ctx_lb = ctx[4] if len(ctx) > 4 else None
        gb5 = getattr(self, "_recsa_gate_b", None)
        if len(ctx) < 5 and gb5 is not None:
            raise NotImplementedError(
                "probe_gate_b is live but this _streamed_ctx producer has no "
                "b carrier")
        if (ctx_lb is None and gb5 is not None
                and min(max(float(gb5), 0.0), 1.0) > 0.0):
            raise RuntimeError(
                "_streamed_ctx bias slot is None but clamp(gate_b) > 0 — "
                "producer dropped a live b")
        q_pre = _decode_q_pre(self, ctx, 5, hidden_states)
        if not _decode_fused_q_norm(ctx, 6):
            q = self.q_norm((self.q_proj(hidden_states) if q_pre is None
                             else q_pre).view(shape))
        else:
            q = rms_norm(self.q_norm, (self.q_proj(hidden_states) if q_pre is None
                                       else q_pre).view(shape), True)
        s_rope_mode = "internal"
        if cfg["mode"] == "external":
            cos, sin = _ext_q_cos_sin(self, q, ctx[7] if len(ctx) > 7 else None,
                                      lambda: positions_from(q0, T, q.device))
            q = apply_rope(q, cos, sin)
            s_rope_mode = "none"
        with torch.cuda.device(q.device):
            out = recsa_forward(
                q.contiguous(), k_full, v_full,
                compression_rate=cfg["c"], max_top_nodes=cfg["mtn"],
                top_k_per_layer=cfg["top_k"], scale=self.scaling,
                rope_base=float(cfg["rope_base"]), rope_mode=s_rope_mode,
                layers_kv=pyramid, q_start=q0,
                force_masks=(cfg["force_masks_by_layer"].get(self.layer_idx)
                             if "force_masks_by_layer" in cfg
                             else cfg.get("force_masks")),
                level_bias=ctx_lb,
                no_summaries=summaries_off(cfg.get("summaries")),
                summary_norm=cfg.get("summary_norm"),
            )
        return self.o_proj(out.reshape(B, T, -1)), None
    # recsa layout is [B, T, H, D]: no transpose
    q = self.q_norm(self.q_proj(hidden_states).view(shape))
    k = self.k_norm(self.k_proj(hidden_states).view(shape))
    v = self.v_proj(hidden_states).view(shape)
    if cfg["mode"] == "external":
        if position_embeddings is None:
            raise RuntimeError("graft mode 'external' needs the decoder layer's position_embeddings "
                               "(true-position RoPE); got None")
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        rope_mode, rope_base = "none", 10000.0
    else:
        rope_mode, rope_base = "internal", float(cfg["rope_base"])
    if cfg["trainable"]:
        no_sum = summaries_off(cfg.get("summaries"))
        if cfg.get("summary_norm") is not None:
            raise RuntimeError("summary_norm is inference-only; the trainable "
                               "graft has no mass_cap support")
        pw = getattr(self, "_recsa_probe_w", None)
        if pw is not None and pw.device != q.device:
            # frozen probe is a plain attribute; model.to() does not move it
            self._recsa_probe_w = pw = pw.to(q.device)
            self._recsa_probe_tau = self._recsa_probe_tau.to(q.device)
        if pw is not None:
            pg = getattr(self, "_recsa_probe_gate", None)
            if pg is not None and pg.device != q.device:
                self._recsa_probe_gate = pg = pg.to(q.device)
            px = getattr(self, "_recsa_probe_extra", None)
            if px is not None and next(iter(px.values())).device != q.device:
                self._recsa_probe_extra = px = {n: t.to(q.device)
                                                for n, t in px.items()}
            with torch.cuda.device(q.device):
                out = recsa_forward_probe_autograd(
                    q, k, v, probe_w=pw, probe_tau=self._recsa_probe_tau,
                    compression_rate=cfg["c"], max_top_nodes=cfg["mtn"],
                    top_k_per_layer=cfg["top_k"], scale=self.scaling,
                    rope_base=rope_base, rope_mode=rope_mode,
                    probe_mode=cfg.get("probe_mode", "per_level"), probe_gate=pg,
                    probe_extra=px, no_summaries=no_sum)
        else:
            with torch.cuda.device(q.device):
                out = recsa_forward_triton_autograd(
                    q, k, v, compression_rate=cfg["c"], max_top_nodes=cfg["mtn"],
                    top_k_per_layer=cfg["top_k"], scale=self.scaling,
                    rope_base=rope_base, rope_mode=rope_mode,
                    no_summaries=no_sum)
        return self.o_proj(out.reshape(B, T, -1)), None
    layers_kv = None
    probe = getattr(self, "_recsa_probe", None)
    if probe is not None:
        pw, pt, pg, px = unpack_probe(probe)
        if pw.device != k.device:
            pw, pt = pw.to(k.device), pt.to(k.device)
            pg = pg.to(k.device) if pg is not None else None
            px = ({n: t.to(k.device) for n, t in px.items()}
                  if px is not None else None)
            self._recsa_probe = (pw, pt, pg, px)
        layers_kv = build_tree_layers(
            k.contiguous(), v.contiguous(), cfg["c"], cfg["mtn"],
            probe_w=pw, probe_tau=pt,
            probe_mode=cfg.get("probe_mode", "per_level"), probe_gate=pg,
            build_chunk_parents=cfg.get("build_chunk_parents"),
            probe_extra=px)
    lb = None
    gb = getattr(self, "_recsa_gate_b", None)
    if gb is not None and layers_kv is not None:
        lb = compose_level_bias(layers_kv[0], T, pw, pt, gb, cfg["c"])
    with torch.cuda.device(q.device):
        out = recsa_forward(
            q.contiguous(), k.contiguous(), v.contiguous(),
            compression_rate=cfg["c"], max_top_nodes=cfg["mtn"],
            top_k_per_layer=cfg["top_k"], scale=self.scaling,
            rope_base=rope_base, rope_mode=rope_mode,
            layers_kv=layers_kv,
            q_chunk=cfg.get("q_chunk"),
            level_bias=lb,
            no_summaries=summaries_off(cfg.get("summaries")),
            summary_norm=cfg.get("summary_norm"),
        )
    return self.o_proj(out.reshape(B, T, -1)), None


_SUMMARIES_UNSET = object()
_SUMMARY_NORM_UNSET = object()


def recorded_summary_norm(model) -> dict | None:
    """The `summary_norm` rule in the recsa_graft.json beside the model's weights (canonical dict), or None."""
    path = (getattr(model, "name_or_path", None)
            or getattr(getattr(model, "config", None), "_name_or_path", None))
    if not path:
        return None
    p = Path(str(path)) / "recsa_graft.json"
    if not p.is_file():
        return None
    rec = json.loads(p.read_text())
    if "summary_norm" not in rec:
        return None
    if rec["summary_norm"] is None:
        raise ValueError(f"{p}: summary_norm recorded as null (an absent key means off)")
    return canonical_summary_norm(rec["summary_norm"])


def recorded_summaries(model) -> str | None:
    path = (getattr(model, "name_or_path", None)
            or getattr(getattr(model, "config", None), "_name_or_path", None))
    if not path:
        return None
    p = Path(str(path)) / "recsa_graft.json"
    if not p.is_file():
        return None
    s = json.loads(p.read_text()).get("summaries", "softmax")
    summaries_off(s)
    return s


def recorded_graft_mode(model) -> str | None:
    path = (getattr(model, "name_or_path", None)
            or getattr(getattr(model, "config", None), "_name_or_path", None))
    if not path:
        return None
    p = Path(str(path)) / "recsa_graft.json"
    if not p.is_file():
        return None
    return check_graft_mode(json.loads(p.read_text()).get("graft_mode", "internal"), str(p))


def probe_trained_graft_mode(ck) -> str:
    rec = ck.get("r2_train_probe") if isinstance(ck, dict) else None
    c = rec.get("config") if isinstance(rec, dict) else None
    if c is None and isinstance(ck, dict) and isinstance(ck.get("config"), dict):
        c = ck["config"]
    return check_graft_mode((c or {}).get("graft_mode") or "internal", "probe checkpoint graft_mode")


def install_recsa_attention(model, *, mode: str, rope_base: float = 10000.0,
                            compression_rate: int = 16, top_k: int = 512,
                            max_top_nodes: int = 8192,
                            probe_ckpt: str | None = None,
                            trainable: bool = False,
                            train_probe: bool = False,
                            q_chunk: int | None = None,
                            build_chunk_parents: int | None = None,
                            summaries=_SUMMARIES_UNSET,
                            summary_norm=_SUMMARY_NORM_UNSET) -> int:
    assert mode in ("internal", "external"), mode
    rec_gm = recorded_graft_mode(model)
    if rec_gm == "external" and mode != "external":
        raise RuntimeError(
            f"{model.name_or_path}/recsa_graft.json records graft_mode='external' (trained with "
            f"true-position RoPE) but install_recsa_attention was called with mode={mode!r} — "
            "refusing to run the candidate-order RoPE operator on it (eval drivers: --graft external)")
    assert max_top_nodes == top_k * compression_rate and top_k % 64 == 0, \
        f"kernel-illegal config top_k={top_k} mtn={max_top_nodes}"
    assert trainable or not train_probe, "train_probe requires trainable=True"
    assert not (trainable and q_chunk), \
        "q_chunk is eval-only; the trainable graft needs the full-T backward context"
    if summaries is _SUMMARIES_UNSET:
        rec_s = recorded_summaries(model)
        if rec_s is not None and rec_s != "softmax":
            raise RuntimeError(
                f"{model.name_or_path}/recsa_graft.json records summaries={rec_s!r} "
                "but install_recsa_attention was called without summaries= (a driver "
                "that does not plumb the switch) — refusing to evaluate the "
                "summaries-on operator on a summaries-off checkpoint; pass "
                "summaries= explicitly (resolve it with graft_geometry.resolve)")
        summaries = "softmax"
    summaries_off(summaries)
    if summary_norm is _SUMMARY_NORM_UNSET:
        rec_sn = recorded_summary_norm(model)
        if rec_sn is not None:
            raise RuntimeError(
                f"{model.name_or_path}/recsa_graft.json records summary_norm={rec_sn!r} "
                "but install_recsa_attention was called without summary_norm= (a driver "
                "that does not plumb the summary_norm switch) — refusing to evaluate the "
                "plain operator on it; pass summary_norm= explicitly (resolve it with "
                "graft_geometry.resolve)")
        summary_norm = None
    summary_norm = canonical_summary_norm(summary_norm)  # ValueError on a bad rule
    if summary_norm is not None and trainable:
        raise RuntimeError("summary_norm is an inference-only operator variant; "
                           "the trainable graft refuses it")
    if summary_norm is not None and summaries_off(summaries):
        raise ValueError("summary_norm with summaries='none': no summary enters the "
                         "softmax, there is nothing to normalise")
    cfg = dict(mode=mode, rope_base=rope_base, c=compression_rate,
               top_k=top_k, mtn=max_top_nodes, trainable=trainable,
               q_chunk=q_chunk, probe_mode="per_level",
               build_chunk_parents=build_chunk_parents,
               summaries=summaries,
               summary_norm=summary_norm)
    probe_w = probe_tau = probe_gate = probe_extra = None
    if probe_ckpt:
        ck = torch.load(probe_ckpt, map_location="cpu", weights_only=False)
        if probe_trained_graft_mode(ck) == "external" and mode != "external":
            raise RuntimeError(
                f"{probe_ckpt} was trained with graft_mode='external' (pools true-position-rotated keys) but "
                f"install_recsa_attention was called with mode={mode!r} — refusing (eval drivers: --graft external)")
        probe_gate_b = None
        if "probe_gate_b" in ck:
            if trainable:
                raise ValueError(
                    f"{probe_ckpt} carries probe_gate_b but the trainable "
                    f"graft has no b carrier — refusing to silently drop it")
            probe_gate_b = ck["probe_gate_b"].float()   # [n_layers]
        probe_w = ck["probe_w"].float()
        probe_tau = ck["probe_tau"].float()  # per_level [n_layers, L_max]; shared_std [n_layers]
        cfg["probe_mode"] = ck.get("probe_mode", "per_level")
        if cfg["probe_mode"] not in {"per_level", "shared_std"} and cfg["probe_mode"] not in MLP_MODES:
            raise ValueError(f"unknown probe_mode {cfg['probe_mode']!r} in {probe_ckpt} "
                             f"(valid: per_level, shared_std, {sorted(MLP_MODES)})")
        if cfg["probe_mode"] == "shared_std" or cfg["probe_mode"] in MLP_MODES:
            assert probe_w.dim() == 2, probe_w.shape
            probe_gate = ck["probe_gate"].float()  # [n_layers]
        if probe_gate_b is not None:
            assert (cfg["probe_mode"] == "shared_std"
                    or cfg["probe_mode"] in MLP_MODES), \
                f"b needs the shared_std α backbone, got {cfg['probe_mode']}"
        if cfg["probe_mode"] in MLP_MODES:
            assert not train_probe, \
                f"{cfg['probe_mode']} has no train_probe path (use probe_kl)"
            # stacked [n_layers, ...] fp32 tensors; sliced per layer below
            probe_extra = {n: t.float() for n, t in ck["probe_extra"].items()}
        assert probe_w.shape[0] == len(model.model.layers), probe_w.shape
    rotary = None
    if mode == "external":
        rotary = getattr(model.model, "rotary_emb", None)
        if rotary is None:
            raise RuntimeError("graft mode 'external' needs model.model.rotary_emb (the model's own RoPE)")
    n = 0
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        assert attn.sliding_window is None, "sliding-window layer unsupported"
        attn._recsa_cfg = cfg
        if rotary is not None:
            object.__setattr__(attn, "_recsa_rotary", rotary)
        if trainable:
            dev = attn.q_proj.weight.device
            if probe_w is not None and train_probe:
                attn.register_parameter("_recsa_probe_w", torch.nn.Parameter(
                    probe_w[i].to(dev), requires_grad=True))
                attn.register_parameter("_recsa_probe_tau", torch.nn.Parameter(
                    probe_tau[i].to(dev), requires_grad=True))
                if probe_gate is not None:
                    attn.register_parameter("_recsa_probe_gate", torch.nn.Parameter(
                        probe_gate[i].to(dev), requires_grad=True))
                else:
                    attn._recsa_probe_gate = None
            elif probe_w is not None:
                attn._recsa_probe_w = probe_w[i].to(dev)
                attn._recsa_probe_tau = probe_tau[i].to(dev)
                attn._recsa_probe_gate = (
                    probe_gate[i].to(dev) if probe_gate is not None else None)
                attn._recsa_probe_extra = (
                    {n: t[i].to(dev) for n, t in probe_extra.items()}
                    if probe_extra is not None else None)
            else:
                attn._recsa_probe_w = attn._recsa_probe_tau = None
                attn._recsa_probe_gate = None
        else:
            attn._recsa_probe = (
                (probe_w[i], probe_tau[i],
                 probe_gate[i] if probe_gate is not None else None,
                 {n: t[i] for n, t in probe_extra.items()}
                 if probe_extra is not None else None)
                if probe_w is not None else None)
            attn._recsa_gate_b = (
                probe_gate_b[i] if probe_ckpt and probe_gate_b is not None
                else None)
        attn.forward = types.MethodType(_recsa_attn_forward, attn)
        n += 1
    if q_chunk:
        install_chunked_mlp(model, int(q_chunk))
    return n


def _chunked_mlp_forward(self, x):
    """Position-wise MLP evaluated in blocks."""
    c = getattr(self, "_recsa_mlp_chunk", None)
    T = x.shape[1]
    if not c or T <= c:
        return self._recsa_mlp_orig(x)
    out = torch.empty_like(x)
    for i in range(0, T, c):
        out[:, i:i + c] = self._recsa_mlp_orig(x[:, i:i + c])
    return out


_chunked_mlp_forward._recsa_chunked_mlp = True


def install_chunked_mlp(model, chunk: int) -> int:
    """Bound the per-layer MLP intermediates to `chunk` positions."""
    n = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        if not hasattr(mlp, "_recsa_mlp_orig"):
            mlp._recsa_mlp_orig = mlp.forward
            mlp.forward = types.MethodType(_chunked_mlp_forward, mlp)
        mlp._recsa_mlp_chunk = int(chunk)
        n += 1
    return n


_STREAM_HOST_ENV = "RECSA_STREAM_HOST"
_STREAM_HOST_MODES = ("pageable", "pinned", "resident", "auto")
_KIB, _GIB = 1 << 10, 1 << 30
_AUTO_CHUNK_BYTES_PER_TOKEN = 160 * _KIB
_AUTO_SEQ_BYTES_PER_TOKEN = 16 * _KIB
_AUTO_MARGIN_MIN = 8 * _GIB
_AUTO_MARGIN_FRAC = 0.10                   # of the device total (14 GiB on an H200)
_PIN_GRANULE_TOKENS = 1 << 16              # pinned cache grows in 64K-token steps
_PREFILL_HN_BUDGET = 1024 << 20


def _stream_host_mode() -> str:
    raw = os.environ.get(_STREAM_HOST_ENV, "")
    mode = raw.strip().lower() or "auto"
    if mode not in _STREAM_HOST_MODES:
        raise ValueError(f"{_STREAM_HOST_ENV}={raw!r}: expected one of "
                         f"{', '.join(_STREAM_HOST_MODES)}")
    return mode


def _stream_mem_info(dev):
    """(driver_free, device_total, allocator_slack) in bytes, or None off-CUDA; slack = reserved - allocated."""
    if dev.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(dev)
    slack = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
    return int(free), int(total), int(max(slack, 0))


def _stream_auto_decision(T: int, hd: int, elem: int, C: int,
                          free: int, total: int, slack: int):
    """auto -> 'resident' iff  need + transient + margin <= free + slack, else 'pinned'."""
    need = T * hd * elem
    transient = (min(C, T) * _AUTO_CHUNK_BYTES_PER_TOKEN
                 + T * _AUTO_SEQ_BYTES_PER_TOKEN)
    margin = max(_AUTO_MARGIN_MIN, int(total * _AUTO_MARGIN_FRAC))
    avail = free + slack
    mode = "resident" if need + transient + margin <= avail else "pinned"
    return mode, dict(need=need, transient=transient, margin=margin,
                      free=free, slack=slack, total=total, avail=avail)


def _stream_pinned_alloc(numel: int, dtype) -> torch.Tensor:
    """Flat page-locked host buffer (recsa.incremental_tree.pinned_host_zeros)."""
    from recsa.incremental_tree import pinned_host_zeros
    return pinned_host_zeros((numel,), dtype)


class _ResidualStream:
    """The streamed forward's [1, T, hidden] residual stream in one of the _STREAM_HOST_MODES. get(s, e) returns rows
    [s, e) on the device; put(s, e, x) stores x there."""

    def __init__(self, model, T: int, hd: int, dt, dev, C: int) -> None:
        requested = _stream_host_mode()
        mode, info = requested, {}
        elem = dt.itemsize
        if requested == "auto":
            mi = _stream_mem_info(dev)
            if mi is None:
                mode, info = "pageable", dict(reason="no device memory info")
            else:
                mode, info = _stream_auto_decision(T, hd, elem, C, *mi)
        self.dev, self.mode, self.buf = dev, mode, None
        if mode == "pageable":
            self.buf = torch.empty(1, T, hd, dtype=dt, device="cpu")
        elif mode == "resident":
            try:
                self.buf = torch.empty(1, T, hd, dtype=dt, device=dev)
            except torch.cuda.OutOfMemoryError:
                if requested != "auto":
                    raise
                mode = self.mode = "pinned"
                info["fallback"] = "resident allocation OOM"
        if mode == "pinned":
            self.buf, info["pin_alloc_s"] = self._pinned(model, T, hd, dt, dev)
        model._recsa_stream_last = dict(requested=requested, mode=mode, T=T,
                                        bytes=T * hd * elem, **info)

    @staticmethod
    def _pinned(model, T, hd, dt, dev):
        numel = T * hd
        cache = getattr(model, "_recsa_stream_pin", None)
        t0 = time.perf_counter()
        fresh = cache is None or cache.dtype != dt or cache.numel() < numel
        if fresh:
            model._recsa_stream_pin = cache = None
            cap = -(-T // _PIN_GRANULE_TOKENS) * _PIN_GRANULE_TOKENS
            with torch.inference_mode(False):
                cache = _stream_pinned_alloc(cap * hd, dt)
            if dev.type == "cuda" and not cache.is_pinned():
                raise RuntimeError("residual-stream host buffer is not page-locked")
            model._recsa_stream_pin = cache
        return (cache[:numel].view(1, T, hd),
                round(time.perf_counter() - t0, 4) if fresh else 0.0)

    def get(self, s: int, e: int) -> torch.Tensor:
        if self.mode == "pageable":
            return self.buf[:, s:e].to(self.dev)
        if self.mode == "pinned":
            return self.buf[:, s:e].to(self.dev, non_blocking=True)
        return self.buf[:, s:e]

    def put(self, s: int, e: int, x: torch.Tensor) -> None:
        if self.mode == "pageable":
            self.buf[:, s:e] = x.to("cpu")
        elif self.mode == "pinned":
            self.buf[:, s:e].copy_(x, non_blocking=True)
        else:
            self.buf[:, s:e].copy_(x)

    def __enter__(self) -> "_ResidualStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close(quiet=exc_type is not None)
        return False

    def close(self, quiet: bool = False) -> None:
        """In pinned mode, synchronize so no async copy still uses the host buffer."""
        if self.mode == "pinned" and self.dev.type == "cuda":
            try:
                torch.cuda.synchronize(self.dev)
            except Exception:
                if not quiet:
                    raise
        self.buf = None


def release_stream_buffers(model) -> None:
    """Free the cached pinned residual buffer (pinned/auto modes)."""
    model._recsa_stream_pin = None


def _streamed_causal_forward(self, input_ids, attention_mask=None,
                             logits_to_keep=0, **kwargs):
    """Streamed-offload forward: residual stream in host memory, only the current layer's K/V and pyramid on the GPU."""
    assert attention_mask is None, "streamed forward takes no attention_mask"
    m = self.model
    dev = m.embed_tokens.weight.device
    dt = m.embed_tokens.weight.dtype
    B, T = input_ids.shape
    assert B == 1, "streamed offload is B=1 (eval cells)"
    C = int(self._streamed_chunk)
    starts = list(range(0, T, C))
    a0 = m.layers[0].self_attn
    hd = self.config.hidden_size
    n_kv, head_dim = self.config.num_key_value_heads, a0.head_dim
    cfg = a0._recsa_cfg
    ext = cfg["mode"] == "external"

    with torch.no_grad(), _ResidualStream(self, T, hd, dt, dev, C) as res:
        k_full = torch.empty(1, T, n_kv, head_dim, dtype=dt, device=dev)
        v_full = torch.empty(1, T, n_kv, head_dim, dtype=dt, device=dev)
        for s in starts:
            e = min(s + C, T)
            res.put(s, e, m.embed_tokens(input_ids[:, s:e].to(dev)))
        spill = getattr(self, "_recsa_decode_spill", None)
        from recsa.decode import (_decoder_layer_tail, _global_hooks_free,
                                      _inline_layer_ok)
        n_keep = max(0, _PREFILL_HN_BUDGET // max(1, min(C, T) * hd * dt.itemsize))
        for li, layer in enumerate(m.layers):
            attn = layer.self_attn
            keep = n_keep if (n_keep and _global_hooks_free()
                              and _inline_layer_ok(layer)) else 0
            hns = {}
            for s in starts:
                e = min(s + C, T)
                hn = layer.input_layernorm(res.get(s, e))
                if not ext:
                    k_full[:, s:e] = attn.k_norm(
                        attn.k_proj(hn).view(1, e - s, -1, head_dim))
                else:
                    kc = attn.k_norm(attn.k_proj(hn).view(1, e - s, -1, head_dim))
                    cos, sin = rope_cos_sin(attn._recsa_rotary, kc,
                                            positions_from(s, e - s, kc.device))
                    k_full[:, s:e] = apply_rope(kc, cos, sin)
                    del kc
                v_full[:, s:e] = attn.v_proj(hn).view(1, e - s, -1, head_dim)
                if len(hns) < keep:
                    hns[s] = hn
                del hn
            probe = getattr(attn, "_recsa_probe", None)
            if probe is not None:
                pw, pt, pg, px = unpack_probe(probe)
                if pw.device != k_full.device:
                    pw, pt = pw.to(k_full.device), pt.to(k_full.device)
                    pg = pg.to(k_full.device) if pg is not None else None
                    px = ({n: t.to(k_full.device) for n, t in px.items()}
                          if px is not None else None)
                    attn._recsa_probe = (pw, pt, pg, px)
                pyramid = build_tree_layers(
                    k_full, v_full, cfg["c"], cfg["mtn"], probe_w=pw,
                    probe_tau=pt, probe_mode=cfg.get("probe_mode", "per_level"),
                    probe_gate=pg,
                    build_chunk_parents=cfg.get("build_chunk_parents"),
                    probe_extra=px)
            else:
                with torch.cuda.device(dev):
                    pyramid = recsa_build_pyramid(k_full, v_full,
                                                  cfg["c"], cfg["mtn"])
            lb = None
            gb = getattr(attn, "_recsa_gate_b", None)
            if gb is not None and probe is not None:
                lb = compose_level_bias(pyramid[0], T, pw, pt, gb, cfg["c"])
            if spill is not None:
                spill(li, k_full, v_full, pyramid)
            for s in starts:
                e = min(s + C, T)
                attn._streamed_ctx = (k_full, v_full, pyramid, s, lb)
                try:
                    if s in hns:
                        out = _decoder_layer_tail(layer, res.get(s, e), hns.pop(s))
                    else:
                        out = layer(res.get(s, e), position_embeddings=None)
                finally:
                    attn._streamed_ctx = None
                res.put(s, e, out[0] if isinstance(out, tuple) else out)
                del out
            del pyramid, hns
        keep = max(1, int(logits_to_keep) or 1)
        tail = m.norm(res.get(T - keep, T))
        logits = self.lm_head(tail)
    from transformers.modeling_outputs import CausalLMOutputWithPast
    return CausalLMOutputWithPast(logits=logits)


def install_streamed_forward(model, *, q_chunk: int) -> int:
    """Install streamed-offload forward."""
    layers = model.model.layers
    for lyr in layers:
        cfg = getattr(lyr.self_attn, "_recsa_cfg", None)
        assert cfg is not None, "install_recsa_attention must run first"
        assert cfg["mode"] in ("internal", "external") and not cfg["trainable"], \
            "streamed offload supports the (internal or external RoPE) eval graft only"
        assert cfg["mode"] == "internal" or getattr(lyr.self_attn, "_recsa_rotary", None) is not None, \
            "external graft without the model's rotary embedding (install_recsa_attention sets it)"
        lyr.self_attn._streamed_ctx = None
    model._streamed_chunk = int(q_chunk)
    model.forward = types.MethodType(_streamed_causal_forward, model)
    return len(layers)
