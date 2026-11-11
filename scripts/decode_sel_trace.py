"""Selection tracing for the eager RecSA decode path: at each decode step, whether the needle's ancestor chain survives
top-k selection at every tree level."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from recsa.decode import RecSADecodeSession
from recsa.naive import _apply_rope, _build_rope_cache


def descent_select(qg: torch.Tensor, ks: Sequence[torch.Tensor], t: int,
                   c: int, top_k: int, cos: torch.Tensor, sin: torch.Tensor,
                   scale: float,
                   force: Optional[Dict[int, torch.Tensor]] = None,
                   ) -> Dict[int, torch.Tensor]:
    """Reference descent (as in naive.py) for one query position, all kv-heads batched. qg [Hkv, G, D] fp32; ks:
    leaves-first level keys, each [1, n_li, Hkv, D]."""
    NL = len(ks)
    Hkv = qg.shape[0]
    device = qg.device
    sel_per_level: Dict[int, torch.Tensor] = {}
    if NL < 2:
        return sel_per_level
    tlp = c ** (NL - 1)
    nvp0 = min(t // tlp // c + 1, top_k)
    vp = torch.arange(nvp0, dtype=torch.long, device=device)[None].expand(Hkv, -1)
    for li in range(NL - 1, 0, -1):
        lk = ks[li][0]                       # [n_li, Hkv, D]
        lp = c ** li
        ri = t // lp
        rci = ri % c
        nvp = vp.shape[1]
        nc = (nvp - 1) * c + rci + 1         # causal clip, as in recsa/naive.py
        rp = nc - 1                          # rightmost candidate, forced
        cp = torch.arange(nc, dtype=torch.long, device=device)
        po = cp // c
        ni = vp[:, po] * c + (cp - po * c)[None]          # [Hkv, nc]
        kn = lk.permute(1, 0, 2).gather(
            1, ni.unsqueeze(-1).expand(-1, -1, lk.shape[-1])).float()
        kr = _apply_rope(kn, cos[cp][None], sin[cp][None])            # [Hkv,nc,D]
        qr = _apply_rope(qg, cos[rp][None, None], sin[rp][None, None]) * scale
        scores = torch.einsum("hgd,hnd->hgn", qr, kr)                 # [Hkv,G,nc]
        lm = scores.amax(dim=(1, 2), keepdim=True)
        imp = torch.exp(scores - lm).sum(1)                           # [Hkv,nc]
        imp[:, rp] = 1e6                     # naive._select_shared_topk forcing
        if force is not None and li in force and force[li].numel():
            imp[torch.isin(ni, force[li].to(ni.device))] = 1e6
        order = torch.sort(imp, dim=-1, descending=True, stable=True).indices
        take = min(top_k, nc)
        sel = ni.gather(1, order[:, :take])                           # [Hkv,take]
        sel_per_level[li] = sel
        vp = torch.sort(sel, dim=-1).values  # sorted parents
    return sel_per_level


class SelectionTrace:
    """Accumulates per-step selection hits for one sample's decode."""

    def __init__(self, spans_tok: List[Tuple[int, int]], *, c: int, top_k: int,
                 theta: float, head_dim: int, device,
                 out_path: Optional[Path] = None,
                 force_nodes: Optional[Dict[int, torch.Tensor]] = None):
        self.spans = [(int(s), int(e)) for s, e in spans_tok if e > s]
        self.force_nodes = force_nodes
        self.c = c
        self.top_k = top_k
        self.cos, self.sin = _build_rope_cache(
            head_dim, top_k * c + 1024, device, theta)
        self.out_path = Path(out_path) if out_path else None
        self.steps: List[dict] = []          # flushed per-step aggregates
        self._acc = None                     # current step accumulator
        self._t = None
        self._nl_seen = set()

    def step_begin(self, t: int) -> None:
        self._flush()
        self._t = t
        self._acc = []

    @torch.inference_mode()
    def on_layer(self, layer_idx: int, attn, hn: torch.Tensor,
                 ks: Sequence[torch.Tensor], t: int) -> None:
        if not self.spans or self._acc is None:
            return
        NL = len(ks)
        self._nl_seen.add(NL)
        if NL < 2:
            return
        hd = attn.head_dim
        q = attn.q_norm(attn.q_proj(hn).view(1, 1, -1, hd))[0, 0].float()
        Hkv = ks[1].shape[2]
        qg = q.view(Hkv, q.shape[0] // Hkv, hd)
        sel = descent_select(qg, ks, t, self.c, self.top_k,
                             self.cos, self.sin, float(attn.scaling),
                             force=self.force_nodes)
        for li, s in sel.items():                       # s: [Hkv, take]
            lp = self.c ** li
            lo = torch.tensor([ts // lp for ts, _ in self.spans],
                              device=s.device)          # [S]
            hi = torch.tensor([(te - 1) // lp for _, te in self.spans],
                              device=s.device)
            inr = (s[None] >= lo[:, None, None]) & (s[None] <= hi[:, None, None])
            hit = inr.any(-1)                           # [S, Hkv] span-any-hit
            any_sum = hit.any(0).sum()                  # 0-dim gpu
            if li == 1:
                vals = [any_sum, hit.float().mean(0).sum(),
                        hit.any(0).float().mean()]
            else:
                vals = [any_sum]
            self._acc.append((layer_idx, li, Hkv, vals))

    def _flush(self) -> None:
        acc = self._acc
        self._acc = None
        if not acc:
            return
        flat = torch.stack([v.float() for rec in acc for v in rec[3]])
        flat = flat.cpu().tolist()                      # single host sync
        anc_hit: Dict[int, float] = {}
        anc_n: Dict[int, int] = {}
        fine_cov_sum = 0.0
        fine_pairs = 0
        fine_by_layer = []
        i = 0
        for layer_idx, li, Hkv, vals in acc:
            any_sum = flat[i]
            i += 1
            if li == 1:
                fine_cov_sum += flat[i]
                fine_pairs += Hkv
                fine_by_layer.append((layer_idx, flat[i + 1]))
                i += 2
            anc_hit[li] = anc_hit.get(li, 0.0) + any_sum
            anc_n[li] = anc_n.get(li, 0) + Hkv
        if fine_pairs == 0:
            return
        self.steps.append({
            "t": self._t,
            "fine_cov": fine_cov_sum / fine_pairs,
            "anc_hit": {str(li): anc_hit[li] / anc_n[li]
                        for li in sorted(anc_hit)},
            "fine_hit_by_layer": [round(v, 4) for _, v in
                                  sorted(fine_by_layer)],
        })

    def summary(self) -> dict:
        self._flush()
        if not self.spans:
            return {"sel_trace": "no_spans"}
        if not self.steps:
            return {"sel_trace": "no_steps", "sel_nl": sorted(self._nl_seen)}
        if self.out_path is not None:
            with open(self.out_path, "w") as f:
                for s in self.steps:
                    f.write(json.dumps(s) + "\n")
        n = len(self.steps)
        covs = [s["fine_cov"] for s in self.steps]
        first16 = covs[:16]
        levels = sorted({li for s in self.steps for li in s["anc_hit"]})
        per_layer = torch.tensor([s["fine_hit_by_layer"] for s in self.steps])
        return {
            "sel_nl": sorted(self._nl_seen),
            "sel_steps": n,
            "sel_n_spans": len(self.spans),
            "sel_fine_cov_mean": round(sum(covs) / n, 4),
            "sel_fine_cov_first16": round(sum(first16) / len(first16), 4),
            "sel_fine_ever": round(float(per_layer.max(0).values.mean()), 4),
            "sel_anc_hit": {li: round(sum(s["anc_hit"][li] for s in self.steps
                                          if li in s["anc_hit"]) /
                                      sum(1 for s in self.steps
                                          if li in s["anc_hit"]), 4)
                            for li in (str(x) for x in levels)},
            "sel_fine_hit_by_layer_mean":
                [round(float(v), 4) for v in per_layer.mean(0)],
        }


class TracingSession(RecSADecodeSession):
    """RecSADecodeSession with a read-only selection trace on the eager step."""

    def __init__(self, model, *, reserve: int = 256,
                 offload_leaves: bool = False,
                 trace: Optional[SelectionTrace] = None) -> None:
        super().__init__(model, reserve=reserve, offload_leaves=offload_leaves)
        self._trace = trace

    def _eager_step(self, token_id: torch.Tensor) -> torch.Tensor:
        m = self.m
        mdl = m.model
        h = mdl.embed_tokens(token_id.to(mdl.embed_tokens.weight.device))
        tr = self._trace
        if tr is not None:
            tr.step_begin(self.t)
        try:
            for li, (layer, st) in enumerate(zip(mdl.layers, self.states)):
                attn = layer.self_attn
                hn = layer.input_layernorm(h)
                hd = attn.head_dim
                st.append(attn.k_norm(attn.k_proj(hn).view(1, 1, -1, hd)),
                          attn.v_proj(hn).view(1, 1, -1, hd))
                ks, vs = st.layers()
                attn._streamed_ctx = (ks[0], vs[0], (ks, vs), self.t,
                                      st.layers_bias())
                try:
                    out = layer(h, position_embeddings=None)
                finally:
                    attn._streamed_ctx = None
                if tr is not None:
                    tr.on_layer(li, attn, hn, ks, self.t)
                h = out[0] if isinstance(out, tuple) else out
            logits = m.lm_head(mdl.norm(h))[:, -1].float()
        except BaseException:
            self._poisoned = True
            raise
        self.t += 1
        return logits


def spans_to_token_ranges(offsets, spans_char: List[Tuple[int, int]],
                          ) -> List[Tuple[int, int]]:
    """Map char spans to token ranges (end-exclusive) via an offsets mapping ([(start_char, end_char)] per token,
    fast-tokenizer convention)."""
    import bisect
    starts = [o[0] for o in offsets]
    ends = [o[1] for o in offsets]
    n = len(offsets)
    out = []
    for s, e in spans_char:
        ts = bisect.bisect_right(ends, s)     # first token ending after s
        te = bisect.bisect_left(starts, e)    # first token starting at/after e
        ts = min(max(ts, 0), n - 1)
        te = max(min(te, n), ts + 1)
        out.append((ts, te))
    return out
