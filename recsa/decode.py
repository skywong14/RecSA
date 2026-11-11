"""Greedy decode for the grafted RecSA model."""

from __future__ import annotations

import gc
import os
import warnings
import weakref
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

import recsa.decode_fused_norm as _dfn
import recsa.incremental_tree as _incr
import recsa.probe_pool as _probe_pool
from recsa.decode_fused_norm import mlp_forward, rms_norm
from recsa.incremental_tree import IncrementalTreeState, OpenParentVBatch
from recsa.parallel import summaries_off
from recsa.probe_pool import MLP_MODES, BucketCursor, compute_num_layers, unpack_probe
from recsa.true_rope import apply_rope, positions_from, rope_cos_sin


def _env_flag(name: str, default: str) -> bool:
    v = os.environ.get(name, default).strip()
    if v not in ("0", "1"):
        raise ValueError(f"{name} must be '0' or '1', got {v!r}")
    return v == "1"


RECSA_DECODE_SINGLE_NORM = _env_flag("RECSA_DECODE_SINGLE_NORM", "1")
RECSA_DECODE_FUSED_QKV = _env_flag("RECSA_DECODE_FUSED_QKV", "1")

RECSA_DECODE_FUSED_NORM = _env_flag("RECSA_DECODE_FUSED_NORM", "1")
RECSA_DECODE_FUSED_SILU = _env_flag("RECSA_DECODE_FUSED_SILU", "1")

RECSA_OFFLOAD_MIRROR = os.environ.get("RECSA_OFFLOAD_MIRROR", "off").strip().lower()
RECSA_OFFLOAD_MIRROR_MARGIN_GIB = float(os.environ.get("RECSA_OFFLOAD_MIRROR_MARGIN_GIB", "8"))


def _parse_mirror_spec(spec: str) -> str:
    s = str(spec).strip().lower()
    try:
        if s in ("off", "auto"):
            return s
        if s.endswith("gib") and float(s[:-3]) >= 0:
            return s
        if s.startswith("frac:") and 0.0 <= float(s[5:]) <= 1.0:
            return s
        if s.startswith("r0:") and int(s[3:]) >= 0:
            return s
    except ValueError:
        pass
    raise ValueError(f"leaf mirror spec must be off | auto | <x>gib | frac:<f> | r0:<n>, got {spec!r}")


_parse_mirror_spec(RECSA_OFFLOAD_MIRROR)  # fail at import on a malformed env value


_MIRRORED_TRANSFORMERS = "5.9.0"
_QWEN3_LAYER_FWD = None  # resolved lazily: the mirrored forward, or False


def _mirrored_layer_forward():
    global _QWEN3_LAYER_FWD
    if _QWEN3_LAYER_FWD is None:
        try:
            import transformers
            from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        except Exception:  # noqa: BLE001
            _QWEN3_LAYER_FWD = False
        else:
            if transformers.__version__ == _MIRRORED_TRANSFORMERS:
                _QWEN3_LAYER_FWD = Qwen3DecoderLayer.forward
            else:
                warnings.warn(
                    f"RECSA_DECODE_SINGLE_NORM: transformers "
                    f"{transformers.__version__} != mirrored "
                    f"{_MIRRORED_TRANSFORMERS}; decode calls the decoder layer "
                    "(two input norms per layer) and runs the separate q/k/v "
                    "projections until the inline tail is re-derived")
                _QWEN3_LAYER_FWD = False
    return _QWEN3_LAYER_FWD


def _plain_call(mod) -> bool:
    """True if mod(...) is plain type(mod).forward(mod, ...), with no instance override or hooks."""
    return not (mod._forward_hooks or mod._forward_pre_hooks
                or "forward" in mod.__dict__)


def _global_hooks_free() -> bool:
    from torch.nn.modules import module as _nnm
    return not (getattr(_nnm, "_global_forward_hooks", None)
                or getattr(_nnm, "_global_forward_pre_hooks", None))


def _inline_layer_ok(layer) -> bool:
    fwd = _mirrored_layer_forward()
    return (bool(fwd) and type(layer).forward is fwd and _plain_call(layer)
            and not (getattr(layer, "gradient_checkpointing", False)
                     and layer.training))


def _decoder_layer_tail(layer, h: torch.Tensor, hn: torch.Tensor,
                        fused_norm: bool = False, fused_silu: bool = False) -> torch.Tensor:
    """Qwen3DecoderLayer.forward of the mirrored transformers version, given hn = layer.input_layernorm(h).
    fused_norm / fused_silu route the post-attention norm and the MLP activation through recsa.decode_fused_norm."""
    a, _ = layer.self_attn(hidden_states=hn, attention_mask=None,
                           position_ids=None, past_key_values=None,
                           use_cache=False, position_embeddings=None)
    h = h + a
    if not (fused_norm or fused_silu):
        return h + layer.mlp(layer.post_attention_layernorm(h))
    return h + mlp_forward(layer.mlp, rms_norm(layer.post_attention_layernorm, h, fused_norm),
                           fused_silu)


def _qkv_projs(attn):
    return attn.q_proj, attn.k_proj, attn.v_proj


def _qkv_fusable(attn) -> bool:
    """fuse_qkv_weights may fuse this layer."""
    ps = _qkv_projs(attn)
    w0 = ps[0].weight
    return all(isinstance(p, torch.nn.Linear)
               and type(p).forward is torch.nn.Linear.forward and _plain_call(p)
               and p.bias is None
               and p.weight.is_cuda and p.weight.dim() == 2
               and p.weight.is_contiguous()
               and p.weight.dtype == w0.dtype and p.weight.device == w0.device
               and p.weight.shape[1] == w0.shape[1] for p in ps) and (
        len({id(p.weight) for p in ps}) == len(ps))


def fused_qkv_weight(attn) -> Optional[torch.Tensor]:
    """The layer's fused [Wq; Wk; Wv] buffer if q/k/v_proj are still row slices of it, else None."""
    W = getattr(attn, "_recsa_wqkv", None)
    if W is None:
        return None
    K, esz, ptr, row = W.shape[1], W.element_size(), W.data_ptr(), 0
    for p in _qkv_projs(attn):
        w = p.weight
        if (type(p).forward is not torch.nn.Linear.forward or p.bias is not None
                or not _plain_call(p) or w.dtype != W.dtype
                or w.device != W.device or w.dim() != 2 or w.shape[1] != K
                or w.stride() != (K, 1) or w.data_ptr() != ptr + row * K * esz):
            return None
        row += w.shape[0]
    return W if row == W.shape[0] else None


_QKV_CANARY_OK: set = set()


def _qkv_canary(W: torch.Tensor, nq: int, nk: int) -> None:
    """Check once per (device, dtype, shape) that the fused M=1 GEMM matches the separate GEMMs on random inputs."""
    key = (str(W.device), W.dtype, tuple(W.shape), nq, nk)
    if key in _QKV_CANARY_OK:
        return
    g = torch.Generator(device=W.device)
    g.manual_seed(20261007)
    with torch.cuda.device(W.device):
        for _ in range(8):
            x = torch.randn(1, 1, W.shape[1], generator=g, device=W.device,
                            dtype=torch.float32).to(W.dtype)
            y = F.linear(x, W)
            for lo, hi in ((0, nq), (nq, nq + nk), (nq + nk, W.shape[0])):
                if not torch.equal(y[..., lo:hi], F.linear(x, W[lo:hi])):
                    raise RuntimeError(
                        f"fused QKV GEMM {tuple(W.shape)} rows [{lo}, {hi}) are "
                        f"not bit-identical to the separate projection on "
                        f"{torch.cuda.get_device_name(W.device)}"
                        " (cuBLAS reduction order changed) — run decode with "
                        "RECSA_DECODE_FUSED_QKV=0")
    _QKV_CANARY_OK.add(key)


def _needs_fusion(model) -> bool:
    return any(fused_qkv_weight(l.self_attn) is None and _qkv_fusable(l.self_attn)
               for l in model.model.layers)


def _live_graph_sessions(model) -> list:
    """Sessions on this model holding a replayable CUDA graph."""
    reg = getattr(model, "_recsa_graph_sessions", None)
    if not reg:
        return []
    live = [s for s in list(reg) if getattr(s, "_graph", None) is not None]
    if live:
        # an unreachable session in a reference cycle can never replay again
        gc.collect()
        live = [s for s in list(reg) if getattr(s, "_graph", None) is not None]
    return live


def fuse_qkv_weights(model) -> int:
    """Re-point every decoder layer's q/k/v projection weights at consecutive row slices of one contiguous [Nq+Nk+Nv,
    hidden] buffer (kept as attn._recsa_wqkv)."""
    if _needs_fusion(model) and _live_graph_sessions(model):
        raise RuntimeError(
            "fuse_qkv_weights: a decode session on this model holds a live CUDA "
            "graph captured on the separate q/k/v weights; fusing would free "
            "the storages that graph replays")
    n = 0
    with torch.inference_mode(False), torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            if fused_qkv_weight(attn) is not None or not _qkv_fusable(attn):
                continue
            ws = [p.weight for p in _qkv_projs(attn)]
            rows = [w.shape[0] for w in ws]
            W = torch.empty(sum(rows), ws[0].shape[1], dtype=ws[0].dtype,
                            device=ws[0].device)
            lo = 0
            for w, r in zip(ws, rows):
                W[lo:lo + r].copy_(w)
                lo += r
            lo = 0
            for w, r in zip(ws, rows):
                w.data = W[lo:lo + r]
                lo += r
            attn._recsa_wqkv = W
            if fused_qkv_weight(attn) is not W:
                raise RuntimeError(
                    f"fuse_qkv_weights: layer {i}: the re-pointed q/k/v weights "
                    "are not recognised as row views of the fused buffer "
                    "(_qkv_fusable and fused_qkv_weight disagree) — run decode "
                    "with RECSA_DECODE_FUSED_QKV=0")
            _qkv_canary(W, rows[0], rows[1])
            n += 1
    if n:
        torch.cuda.empty_cache()
    return n


class RecSADecodeSession:
    """B=1 greedy decode on a model set up by install_recsa_attention and install_streamed_forward."""

    def __init__(self, model, *, reserve: int = 256,
                 offload_leaves: bool = False,
                 leaf_mirror: Optional[str] = None) -> None:
        self.m = model
        self.reserve = int(reserve)
        self.offload_leaves = bool(offload_leaves)
        if leaf_mirror is not None and not self.offload_leaves:
            raise ValueError("leaf_mirror needs offload_leaves=True")
        self._mirror_spec = (_parse_mirror_spec(leaf_mirror) if leaf_mirror is not None
                             else _parse_mirror_spec(RECSA_OFFLOAD_MIRROR)
                             if self.offload_leaves else "off")
        self.mirror_info: Optional[dict] = None
        self.states: List[IncrementalTreeState] = []
        self.t = 0  # tokens currently in context
        self._poisoned = False  # set if a step() died mid-layer-loop
        a0 = model.model.layers[0].self_attn
        cfg = getattr(a0, "_recsa_cfg", None)
        if cfg is None or cfg["mode"] not in ("internal", "external") or cfg["trainable"]:
            raise ValueError("decode needs the internal-rope eval graft (or the external-RoPE one, graft "
                             "mode 'external'), non-trainable")
        self._ext = cfg["mode"] == "external"
        if self._ext and getattr(a0, "_recsa_rotary", None) is None:
            raise ValueError("external graft without the model's rotary embedding")
        if not hasattr(model, "_streamed_chunk"):
            raise ValueError("decode prefill needs install_streamed_forward")
        self.cfg = cfg

        self.summaries = "none" if summaries_off(cfg.get("summaries")) else "softmax"
        self.summary_norm = cfg.get("summary_norm")
        self._single_norm = bool(RECSA_DECODE_SINGLE_NORM)
        self._fused_qkv = bool(RECSA_DECODE_FUSED_QKV)
        # RECSA_DECODE_FUSED_NORM / RECSA_DECODE_FUSED_SILU: fixed for the session
        self._fused_norm = bool(RECSA_DECODE_FUSED_NORM)
        self._fused_silu = bool(RECSA_DECODE_FUSED_SILU)
        if self._fused_qkv and _needs_fusion(model):
            if _live_graph_sessions(model):
                warnings.warn(
                    "RECSA_DECODE_FUSED_QKV: not fusing the q/k/v weights while "
                    "another decode session on this model holds a live CUDA "
                    "graph captured on them; this session runs the separate "
                    "projections")
            else:
                fuse_qkv_weights(model)
        self.a2_last_step = (0, 0)
        # fused RMSNorm / silu*mul call counts of the last step
        self.a9_last_step = (0, 0)

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """One streamed forward over the prompt; capture per-layer trees."""
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"B=1 [1, T] input_ids required, got {tuple(input_ids.shape)}")
        m = self.m
        self.states = []
        self.t = -1
        captured: List[IncrementalTreeState] = []

        a0 = m.model.layers[0].self_attn
        pool = IncrementalTreeState.preallocate(
            len(m.model.layers), int(input_ids.shape[1]),
            self.cfg["c"], self.cfg["mtn"],
            b=1, h_kv=a0.k_proj.out_features // a0.head_dim,
            k_dim=a0.head_dim, v_dim=a0.head_dim,
            dtype=a0.k_proj.weight.dtype, device=a0.k_proj.weight.device,
            reserve=self.reserve, offload_leaves=self.offload_leaves)

        def spill(li: int, k_full, v_full, pyramid) -> None:
            if li != len(captured):
                raise RuntimeError(
                    f"spill for layer {li} but {len(captured)} layers captured "
                    "— out-of-order or duplicate spill callback")
            attn = m.model.layers[li].self_attn
            pw, pt, pg, px = unpack_probe(getattr(attn, "_recsa_probe", None))
            gb = getattr(attn, "_recsa_gate_b", None)
            captured.append(IncrementalTreeState.from_layers(
                pyramid[0], pyramid[1], self.cfg["c"], self.cfg["mtn"],
                probe_w=pw, probe_tau=pt,
                probe_mode=self.cfg.get("probe_mode", "per_level"),
                probe_gate=pg, probe_extra=px,
                reserve=self.reserve, buffers=pool[li],
                gate_b=float(gb) if gb is not None else None))

        m._recsa_decode_spill = spill
        try:
            out = m(input_ids, logits_to_keep=1)
        finally:
            m._recsa_decode_spill = None
        if len(captured) != len(m.model.layers):
            raise RuntimeError(
                f"captured {len(captured)}/{len(m.model.layers)} layers — "
                "was the streamed forward actually used?")
        if self._mirror_spec != "off":
            self.mirror_info = self._attach_leaf_mirror(captured, int(input_ids.shape[1]))
        self.states = captured
        self.t = input_ids.shape[1]
        self._poisoned = False
        return out.logits[:, -1].float()

    def _attach_leaf_mirror(self, states, n_prompt: int) -> dict:
        """Size R0 of the leaf mirror from the session's spec and attach the same mirror range to every
        layer."""
        spec = self._mirror_spec
        st0 = states[0]
        c = st0.c
        cap_rows = st0._k[0].shape[1]
        row_bytes = sum(st._k[0][0, 0].numel() * st._k[0].element_size()
                        + st._v[0][0, 0].numel() * st._v[0].element_size() for st in states)
        info = dict(spec=spec, cap_rows=cap_rows, row_bytes=row_bytes, n_prompt=n_prompt)
        if spec.startswith("r0:"):
            r0 = int(spec[3:])
        elif spec.startswith("frac:"):
            r0 = int((1.0 - float(spec[5:])) * n_prompt) // c * c
        else:
            dev = st0._dev
            if spec == "auto":
                torch.cuda.synchronize(dev)
                torch.cuda.empty_cache()
                free = torch.cuda.mem_get_info(dev)[0]
                budget = free - RECSA_OFFLOAD_MIRROR_MARGIN_GIB * 2 ** 30
                info.update(free_gib=round(free / 2 ** 30, 2),
                            margin_gib=RECSA_OFFLOAD_MIRROR_MARGIN_GIB)
            else:
                budget = float(spec[:-3]) * 2 ** 30
            rows = int(max(0.0, budget) // row_bytes)
            r0 = max(0, cap_rows - rows)
        r0 = -(-r0 // c) * c  # up to a bucket boundary
        if r0 >= cap_rows:
            info.update(r0=None, rows=0, gib=0.0, prompt_fraction=0.0)
            return info
        for st in states:
            st.attach_leaf_mirror(r0)
        info.update(r0=r0, rows=cap_rows - r0,
                    gib=round((cap_rows - r0) * row_bytes / 2 ** 30, 3),
                    prompt_fraction=round(max(0, n_prompt - r0) / n_prompt, 4))
        return info

    @torch.inference_mode()
    def step(self, token_id: torch.Tensor) -> torch.Tensor:
        """Append one token (long [1, 1]) and return its logits [1, vocab]."""
        if self._poisoned:
            raise RuntimeError("session poisoned by an earlier failed step(); re-prefill")
        if self.t < 0 or len(self.states) != len(self.m.model.layers):
            raise RuntimeError(
                f"{len(self.states)} captured states for {len(self.m.model.layers)} "
                f"layers (t={self.t}) — prefill() did not complete")
        if getattr(self, "_graph_on", False):
            return self._graph_step(token_id)
        return self._eager_step(token_id)

    def _layer_kv(self, attn, hn, hd, hooks_free, layer):
        """This token's (k_t, v_t, q_pre) for one layer."""
        W = (fused_qkv_weight(attn)
             if (self._fused_qkv and hooks_free and _inline_layer_ok(layer)
                 and not attn._forward_pre_hooks) else None)
        fn = self._fused_norm
        if W is None:
            if not fn:
                return (attn.k_norm(attn.k_proj(hn).view(1, 1, -1, hd)),
                        attn.v_proj(hn).view(1, 1, -1, hd), None)
            return (rms_norm(attn.k_norm, attn.k_proj(hn).view(1, 1, -1, hd), True),
                    attn.v_proj(hn).view(1, 1, -1, hd), None)
        nq = attn.q_proj.weight.shape[0]
        nk = attn.k_proj.weight.shape[0]
        qkv = F.linear(hn, W)
        if not fn:
            return (attn.k_norm(qkv[..., nq:nq + nk].view(1, 1, -1, hd)),
                    qkv[..., nq + nk:].view(1, 1, -1, hd), qkv[..., :nq])
        return (rms_norm(attn.k_norm, qkv[..., nq:nq + nk].view(1, 1, -1, hd), True),
                qkv[..., nq + nk:].view(1, 1, -1, hd), qkv[..., :nq])

    def _norm(self, mod, x):
        """mod(x) for the input / final RMSNorm, fused under RECSA_DECODE_FUSED_NORM."""
        return rms_norm(mod, x, True) if self._fused_norm else mod(x)

    def _eager_step(self, token_id: torch.Tensor) -> torch.Tensor:
        m = self.m
        mdl = m.model
        h = mdl.embed_tokens(token_id.to(mdl.embed_tokens.weight.device))
        try:
            gh = _global_hooks_free()
            n_in = n_fq = 0
            c0 = dict(_dfn.CALLS)
            ext_cs = None
            for layer, st in zip(mdl.layers, self.states):
                attn = layer.self_attn
                hn = self._norm(layer.input_layernorm, h)
                hd = attn.head_dim
                k_t, v_t, q_pre = self._layer_kv(attn, hn, hd, gh, layer)
                if self._ext:
                    if ext_cs is None:
                        ext_cs = rope_cos_sin(attn._recsa_rotary, k_t,
                                              positions_from(self.t, 1, k_t.device))
                    k_t = apply_rope(k_t, *ext_cs)
                st.append(k_t, v_t)
                ks, vs = st.layers()
                ctx = (ks[0], vs[0], (ks, vs), self.t, st.layers_bias())
                if self._ext:
                    attn._streamed_ctx = ctx + (q_pre, self._fused_norm, ext_cs)
                elif self._fused_norm:
                    attn._streamed_ctx = ctx + (q_pre, True)
                else:
                    attn._streamed_ctx = ctx if q_pre is None else ctx + (q_pre,)
                n_fq += q_pre is not None
                try:
                    if self._single_norm and gh and _inline_layer_ok(layer):
                        h = _decoder_layer_tail(layer, h, hn, self._fused_norm,
                                                self._fused_silu)
                        n_in += 1
                    else:
                        out = layer(h, position_embeddings=None)
                        h = out[0] if isinstance(out, tuple) else out
                finally:
                    attn._streamed_ctx = None
            self.a2_last_step = (n_in, n_fq)
            logits = m.lm_head(self._norm(mdl.norm, h))[:, -1].float()
            self.a9_last_step = tuple(_dfn.CALLS[k] - c0[k] for k in ("norm", "silu"))
        except BaseException:
            self._poisoned = True
            raise
        self.t += 1
        return logits

    # -- CUDA-graph decode ---------------------------------

    def enable_cuda_graph(self) -> None:
        """Whole-step CUDA graph capture/replay."""
        if self.t < 0 or not self.states:
            raise RuntimeError("enable_cuda_graph() requires a completed prefill")
        if any(getattr(st, "_g_b", 0.0) > 0.0 for st in self.states):
            raise NotImplementedError(
                "probe_gate_b is live: the CUDA-graph decode session has no "
                "b carrier — use the eager session (the graph session has "
                "no bias carrier)")
        # top level, not leaves: offloaded leaves live on the host
        st0 = self.states[0]
        dev = st0._k[-1].device
        self._t_dev = torch.tensor([self.t], dtype=torch.int64, device=dev)
        mlp = st0._probe_w is not None and st0._mode in MLP_MODES

        v2 = (mlp and _probe_pool.RECSA_MLP_CURSOR_POOL == "v2"
              and (not st0._offload or _incr.OFFLOAD_POOL_V2 == "on"))

        exact = v2 and _probe_pool.RECSA_MLP_POOL_SEGMENTS in ("exact", "fused")
        fused = exact and _probe_pool.RECSA_MLP_POOL_SEGMENTS == "fused"
        self._g_win = (BucketCursor(self._t_dev, st0.c, st0._k[0].shape[3],
                                    st0._capacity, fused=v2)
                       if mlp else None)
        for st in self.states:
            st.bind_cursor(self._t_dev, self._g_win, defer_v=v2, exact=exact,
                           fused=fused)
        self._g_vb = OpenParentVBatch(self.states, self._g_win) if v2 else None
        if self._g_vb is not None:
            self._g_vb.prepare()
        self._graph_on = True
        self._graph = None
        self._graph_pool = None
        self._g_warm = 0
        self._g_side = torch.cuda.Stream(device=dev)
        self._g_tok = torch.empty(1, 1, dtype=torch.long, device=dev)
        self._g_logits: Optional[torch.Tensor] = None

    def _cursor_body(self) -> torch.Tensor:
        """One decode step that reads every step-varying scalar from the device cursor; the graph capture body, also run
        eagerly for warmup."""
        m = self.m
        mdl = m.model
        if self._g_win is not None:
            self._g_win.refresh(self.states[0].num_layers)
        h = mdl.embed_tokens(self._g_tok)
        gh = _global_hooks_free()
        n_in = n_fq = 0
        c0 = dict(_dfn.CALLS)
        ext_cs = None
        for layer, st in zip(mdl.layers, self.states):
            attn = layer.self_attn
            hn = self._norm(layer.input_layernorm, h)
            hd = attn.head_dim
            k_t, v_t, q_pre = self._layer_kv(attn, hn, hd, gh, layer)
            if self._ext:
                if ext_cs is None:
                    ext_cs = rope_cos_sin(attn._recsa_rotary, k_t, self._t_dev.view(1, 1))
                k_t = apply_rope(k_t, *ext_cs)
            st.append_cursor(k_t, v_t)
            if self._ext:
                attn._graph_ctx = (st, self.cfg, q_pre, self._fused_norm, ext_cs)
            elif self._fused_norm:
                attn._graph_ctx = (st, self.cfg, q_pre, True)
            else:
                attn._graph_ctx = (st, self.cfg) if q_pre is None else (st, self.cfg, q_pre)
            n_fq += q_pre is not None
            try:
                if self._single_norm and gh and _inline_layer_ok(layer):
                    h = _decoder_layer_tail(layer, h, hn, self._fused_norm,
                                            self._fused_silu)
                    n_in += 1
                else:
                    out = layer(h, position_embeddings=None)
                    h = out[0] if isinstance(out, tuple) else out
            finally:
                attn._graph_ctx = None
        self.a2_last_step = (n_in, n_fq)
        if self._g_vb is not None:
            self._g_vb()
        logits = m.lm_head(self._norm(mdl.norm, h))[:, -1].float()
        self.a9_last_step = tuple(_dfn.CALLS[k] - c0[k] for k in ("norm", "silu"))
        self._t_dev.add_(1)
        return logits

    def _graph_step(self, token_id: torch.Tensor) -> torch.Tensor:
        st0 = self.states[0]
        if self.t + 1 > st0._capacity:
            # the replayed sequence cannot check this; refuse on the host
            raise RuntimeError(
                f"reserve exhausted: {self.t} + 1 > capacity {st0._capacity}")
        if compute_num_layers(self.t + 1, st0.c, st0.mtn) != st0.num_layers:
            for st in self.states:
                st.sync_sizes(self.t)
            if self._graph is not None:
                self._g_retired = getattr(self, "_g_retired", [])
                self._g_retired.append((self._graph, self._g_logits))
                self._graph = None
            self._g_warm = min(self._g_warm, 1)
            logits = self._eager_step(token_id)
            self._t_dev.fill_(self.t)
            return logits
        try:
            if self._graph is None:
                self._g_tok.copy_(token_id)
                if self._g_warm < 2:
                    self._g_side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self._g_side):
                        logits = self._cursor_body()
                    torch.cuda.current_stream().wait_stream(self._g_side)
                    self._g_warm += 1
                    self.t += 1
                    return logits
                g = torch.cuda.CUDAGraph()
                if self._graph_pool is None:
                    with torch.cuda.graph(g):
                        self._g_logits = self._cursor_body()
                    self._graph_pool = g.pool()
                else:
                    with torch.cuda.graph(g, pool=self._graph_pool):
                        self._g_logits = self._cursor_body()
                g.replay()  # capture does not execute; replay runs this step
                self._graph = g
                reg = getattr(self.m, "_recsa_graph_sessions", None)
                if reg is None:
                    reg = self.m._recsa_graph_sessions = weakref.WeakSet()
                reg.add(self)
                self.t += 1
                return self._g_logits.clone()
            self._g_tok.copy_(token_id)
            self._graph.replay()
            self.t += 1
            return self._g_logits.clone()
        except BaseException:
            self._poisoned = True
            raise

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, *,
                 eos_token_ids: Optional[Sequence[int]] = None,
                 use_cuda_graph: bool = False) -> List[int]:
        """Greedy decode; returns generated token ids (eos included if hit). use_cuda_graph=True replays a captured
        step graph after the prefill."""
        if max_new_tokens > self.reserve:
            raise ValueError(
                f"max_new_tokens {max_new_tokens} > reserve {self.reserve}")
        eos = {int(e) for e in (eos_token_ids or ())}
        logits = self.prefill(input_ids)
        if use_cuda_graph:
            self.enable_cuda_graph()
        out: List[int] = []
        tok = torch.empty(1, 1, dtype=torch.long, device=input_ids.device)
        n = int(max_new_tokens)
        for i in range(n):
            nxt = int(logits.argmax(-1).item())
            out.append(nxt)
            if nxt in eos or i == n - 1:  # the final token's step would be discarded
                break
            tok.fill_(nxt)
            logits = self.step(tok)
        if use_cuda_graph:

            for st in self.states:
                st.sync_sizes(self.t)
        return out
