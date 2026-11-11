"""Persist the RecSA graft geometry with the weights, and enforce it at eval."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

SIDECAR = "recsa_graft.json"
SCHEMA = 1

#: branching factor (fixed)
COMPRESSION_RATE = 16
#: top_k values that may be run; kernel requires top_k % 64 == 0
ALLOWED_TOP_K = (64, 128, 256, 512, 1024)
DEFAULT_TOP_K = 512
#: operator variants (recsa/parallel.py SUMMARIES_MODES); absent key = "softmax"
SUMMARIES = ("softmax", "none")
DEFAULT_SUMMARIES = "softmax"
#: resolve() sentinel: the caller does not pass summaries
_UNAWARE = object()
SUMMARY_NORM_RULES = ("mass_cap",)
SUMMARY_NORM_ARCHIVED_RULES = ("count_clamp", "static", "equalize_clamp", "combo")
SUMMARY_NORM_ARCHIVE_TAG = "not part of this package"
#: RoPE modes of the graft (src/true_rope.GRAFT_MODES); absent key = "internal"
GRAFT_MODES = ("internal", "external")
DEFAULT_GRAFT_MODE = "internal"


def summary_norm_archived_message(rule) -> str:
    return f"rule {rule!r}: {SUMMARY_NORM_ARCHIVE_TAG}"

def _check_top_k(k: int, where: str) -> int:
    if k not in ALLOWED_TOP_K:
        raise SystemExit(
            f"{where}: top_k={k} is not permitted. Allowed: "
            f"{', '.join(map(str, ALLOWED_TOP_K))} (default {DEFAULT_TOP_K}, "
            "the trained value).")
    return int(k)


def _check_summaries(s, where: str) -> str:
    if s not in SUMMARIES:
        raise SystemExit(
            f"{where}: summaries={s!r} is not a known operator variant. "
            f"Allowed: {', '.join(SUMMARIES)} (absent = {DEFAULT_SUMMARIES}).")
    return s


def _check_graft_mode(gm, where: str) -> str:
    if gm not in GRAFT_MODES:
        raise SystemExit(
            f"{where}: graft_mode={gm!r} is not a known RoPE mode. "
            f"Allowed: {', '.join(GRAFT_MODES)} (absent = {DEFAULT_GRAFT_MODE}).")
    return gm


def _check_summary_norm(sn, where: str) -> dict:
    """Canonical dict of a summary_norm rule; SystemExit on a bad rule."""
    def bad(msg):
        raise SystemExit(f"{where}: summary_norm={sn!r} is not a valid rule ({msg}). "
                         "Allowed: {'rule': 'mass_cap', 'tau': <0 < tau < 1>[, 'n_ref': <int >= 1>]} "
                         "(absent = off).")
    if not isinstance(sn, dict):
        bad("not a dict")
    rule = sn.get("rule")
    if rule == "mass_cap":
        if not {"rule", "tau"} <= set(sn) <= {"rule", "tau", "n_ref"}:
            bad("mass_cap takes the keys rule, tau and optionally n_ref (the gate); it is not "
                "combinable with any other rule (count_clamp, static, equalize_clamp and combo are "
                f"{SUMMARY_NORM_ARCHIVE_TAG})")
        tau = sn["tau"]
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) \
                or not 0.0 < tau < 1.0:
            bad("mass_cap tau must be a number with 0 < tau < 1")
        out = dict(rule="mass_cap", tau=float(tau))
        if "n_ref" in sn:
            gate = sn["n_ref"]
            if isinstance(gate, bool) or not isinstance(gate, int) or gate < 1:
                bad("mass_cap n_ref (the gate) must be an int >= 1")
            out["n_ref"] = int(gate)
        return out
    if rule in SUMMARY_NORM_ARCHIVED_RULES:
        bad(summary_norm_archived_message(rule))
    bad(f"rule must be one of {SUMMARY_NORM_RULES}")


_SN_CLI_GRAMMAR = "none | mass_cap:<tau>[:n_ref=<int>]"
_SN_CLI_PLUS = re.compile(r"\+(?=[A-Za-z_]+:)")


def parse_summary_norm(s: str):
    s = str(s).strip()
    if s == "none":
        return "none"
    try:
        parts = _SN_CLI_PLUS.split(s)
        if len(parts) > 1:
            heads = [p.partition(":")[0] for p in parts]
            if sorted(heads) == ["count_clamp", "mass_cap"]:
                # the refused combo <count_clamp>+<mass_cap> (either order)
                raise ValueError(summary_norm_archived_message("combo"))
            archived = [h for h in heads if h in SUMMARY_NORM_ARCHIVED_RULES]
            if archived:
                raise ValueError(summary_norm_archived_message(archived[0]))
            raise ValueError(f"{_SN_CLI_GRAMMAR}; one rule, no '+' combination "
                             f"(got {'+'.join(heads)})")
        return _parse_one_rule(s)
    except (ValueError, SystemExit) as e:
        raise argparse.ArgumentTypeError(f"bad --summary-norm {s!r}: {e}") from None


def _parse_one_rule(s: str) -> dict:
    """Parse one rule (no '+') into its canonical dict."""
    head, _, rest = s.partition(":")
    if head == "mass_cap":
        tau, *fields = rest.split(":") if rest else [""]
        sn = dict(rule="mass_cap")
        if not tau or "," in tau:
            raise ValueError("mass_cap:<tau>[:n_ref=<int>] takes one number, 0 < tau < 1")
        sn["tau"] = float(tau)
        for f in fields:
            name, eq, val = f.partition("=")
            if name != "n_ref" or not eq or "n_ref" in sn or not val.isdigit():
                raise ValueError("mass_cap:<tau>[:n_ref=<int>]: the only suffix field is the "
                                 "gate n_ref=<int>, at most once (mass_cap is not combinable "
                                 "with any other rule; count_clamp, static, equalize_clamp and "
                                 f"combo are {SUMMARY_NORM_ARCHIVE_TAG})")
            sn["n_ref"] = int(val)
        return _check_summary_norm(sn, "--summary-norm")
    if head in SUMMARY_NORM_ARCHIVED_RULES:
        raise ValueError(summary_norm_archived_message(head))
    raise ValueError(_SN_CLI_GRAMMAR)


def _tag_num(x: float) -> str:
    """Label-safe number: shortest repr without a trailing '.0', '-' -> 'm', '.' -> 'p' (0.3 -> 0p3)."""
    r = repr(float(x))
    if r.endswith(".0"):
        r = r[:-2]
    return r.replace("-", "m").replace(".", "p")


def summary_norm_tag(sn) -> str:
    if sn is None:
        return ""
    sn = _check_summary_norm(sn, "summary_norm_tag")
    return "mc" + _tag_num(sn["tau"]) + (f"_n{sn['n_ref']}" if "n_ref" in sn else "")


def write(export_dir, *, top_k: int, source: str,
          compression_rate: int = COMPRESSION_RATE, summaries: str | None = None,
          summary_norm: dict | None = None, graft_mode: str | None = None,
          **context) -> Path:
    """Record the graft geometry alongside an exported checkpoint."""
    if compression_rate != COMPRESSION_RATE:
        raise SystemExit(
            f"compression_rate={compression_rate} but {COMPRESSION_RATE} is a "
            "hard invariant of this pipeline — refusing to record it.")
    _check_top_k(top_k, "write")
    extra = {} if summaries is None else dict(summaries=_check_summaries(summaries, "write"))
    if summary_norm is not None:
        extra["summary_norm"] = _check_summary_norm(summary_norm, "write")
    if graft_mode is not None and _check_graft_mode(graft_mode, "write") != DEFAULT_GRAFT_MODE:
        extra["graft_mode"] = graft_mode
    rec = dict(schema=SCHEMA, compression_rate=COMPRESSION_RATE,
               top_k=int(top_k), max_top_nodes=int(top_k) * COMPRESSION_RATE,
               **extra, source=source, **context)
    p = Path(export_dir) / SIDECAR
    p.write_text(json.dumps(rec, indent=2) + "\n")
    return p


def load(export_dir) -> dict | None:
    p = Path(export_dir) / SIDECAR
    if not p.is_file():
        return None
    rec = json.loads(p.read_text())
    if rec.get("schema") != SCHEMA:
        raise SystemExit(
            f"{p}: schema {rec.get('schema')} != {SCHEMA} — refusing to guess")
    if rec["compression_rate"] != COMPRESSION_RATE:
        raise SystemExit(
            f"{p}: compression_rate={rec['compression_rate']} but "
            f"{COMPRESSION_RATE} is a hard invariant — this export was built by "
            "an incompatible pipeline; refusing to run rather than coerce it.")
    if rec["max_top_nodes"] != rec["top_k"] * COMPRESSION_RATE:
        raise SystemExit(
            f"{p}: max_top_nodes {rec['max_top_nodes']} != top_k*c "
            f"{rec['top_k'] * COMPRESSION_RATE} — corrupt geometry record")
    rec["summaries"] = _check_summaries(rec.get("summaries", DEFAULT_SUMMARIES), str(p))
    rec["summary_norm"] = (_check_summary_norm(rec["summary_norm"], str(p))
                           if "summary_norm" in rec else None)
    rec["graft_mode"] = _check_graft_mode(rec.get("graft_mode", DEFAULT_GRAFT_MODE), str(p))
    return rec


def resolve(export_dir, *, top_k: int | None, summaries=_UNAWARE, log=print,
            summary_norm=_UNAWARE) -> dict:
    """Resolve the geometry to run with."""
    aware = summaries is not _UNAWARE
    want_s = None if not aware else summaries
    if want_s is not None:
        _check_summaries(want_s, "cli")
    sn_aware = summary_norm is not _UNAWARE
    want_sn = None if not sn_aware else summary_norm
    if isinstance(want_sn, str):
        want_sn = parse_summary_norm(want_sn) if want_sn != "none" else "none"
    elif want_sn is not None:
        want_sn = _check_summary_norm(want_sn, "cli")
    rec = load(export_dir)
    if rec is None:
        eff_k = _check_top_k(DEFAULT_TOP_K if top_k is None else top_k, "cli")
        eff_s = DEFAULT_SUMMARIES if want_s is None else want_s
        eff_sn = None if want_sn in (None, "none") else want_sn
        _check_sn_vs_summaries(eff_sn, eff_s, export_dir)
        log("!" * 72)
        log(f"WARNING: no {SIDECAR} in {export_dir} — graft geometry CANNOT be "
            "verified against training.")
        log(f"         running c={COMPRESSION_RATE} top_k={eff_k} "
            f"mtn={eff_k * COMPRESSION_RATE} on the caller's word alone.")
        log("         The evaluated top_k may then differ from "
            "the trained one.")
        log("!" * 72)
        return dict(compression_rate=COMPRESSION_RATE, top_k=eff_k,
                    max_top_nodes=eff_k * COMPRESSION_RATE, summaries=eff_s,
                    summary_norm=eff_sn,
                    verified=False, recorded=None, off_config=None)

    if not aware and rec["summaries"] != DEFAULT_SUMMARIES:
        raise SystemExit(
            f"{export_dir}: {SIDECAR} records summaries={rec['summaries']!r}, but "
            "this caller does not plumb the summaries switch (graft_geometry.resolve "
            "called without summaries=) — refusing to evaluate the summaries-on "
            "operator on a summaries-off checkpoint.")
    if not sn_aware and rec["summary_norm"] is not None:
        raise SystemExit(
            f"{export_dir}: {SIDECAR} records summary_norm={rec['summary_norm']!r}, but "
            "this caller does not plumb the summary_norm switch (graft_geometry.resolve "
            "called without summary_norm=) — refusing to evaluate the plain operator "
            "on it.")
    eff_k = _check_top_k(rec["top_k"] if top_k is None else top_k, "cli")
    eff_s = rec["summaries"] if want_s is None else want_s
    eff_sn = (rec["summary_norm"] if want_sn is None
              else None if want_sn == "none" else want_sn)
    _check_sn_vs_summaries(eff_sn, eff_s, export_dir)
    off_k = eff_k != rec["top_k"]
    off_s = eff_s != rec["summaries"]
    off_sn = eff_sn != rec["summary_norm"]
    if off_k:
        log(f"[graft geometry] top_k={eff_k} differs from the trained "
            f"{rec['top_k']} — deliberate off-config run, recorded in summary.")
    if off_s:
        log(f"[graft geometry] summaries={eff_s!r} differs from the trained "
            f"{rec['summaries']!r} — deliberate off-config run, recorded in summary.")
    if off_sn:
        log(f"[graft geometry] summary_norm={eff_sn!r} differs from the recorded "
            f"{rec['summary_norm']!r} — deliberate off-config run, recorded in summary.")
    gm = {} if rec["graft_mode"] == DEFAULT_GRAFT_MODE else {"graft_mode": rec["graft_mode"]}
    return dict(compression_rate=COMPRESSION_RATE, top_k=eff_k,
                max_top_nodes=eff_k * COMPRESSION_RATE, summaries=eff_s,
                summary_norm=eff_sn, **gm,
                verified=True, recorded={**{k: rec[k] for k in
                                            ("compression_rate", "top_k", "max_top_nodes",
                                             "summaries", "summary_norm")}, **gm},
                off_config=off_k or off_s or off_sn)


def _check_sn_vs_summaries(eff_sn, eff_s, export_dir) -> None:
    if eff_sn is not None and eff_s == "none":
        raise SystemExit(
            f"{export_dir}: summary_norm={eff_sn!r} with summaries='none' — no summary "
            "enters the softmax, there is nothing to normalise; refusing the combination.")
