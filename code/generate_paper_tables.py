#!/usr/bin/env python3
"""
generate_paper_tables.py

Generates two LaTeX tables for the paper:

  Table 1 (table_results_main.tex):
    Comprehensive results — Mixture / Local / Graph-F / Graph-All for IRM, PSM, cIRM
    Rows: Overall (Standard, Worst), Speaker (1/2/3), SNR (Easy/Med/Hard), RT60 (Dry/Mild/Wet)
    Values: mean output SI-SDR (dB).  Mixture column = mean input SI-SDR.

  Table 2 (table_ablation_arch.tex):
    Architecture ablation — Local vs. Bottleneck-only (V7) vs. Input+Bott (V8=Graph-F)
    Shows contribution of each graph injection mechanism.

Tier definitions
  SNR:  Easy ≥10 dB,  Medium 0–10 dB,  Hard <0 dB
  RT60: Dry <0.5 s,   Mild 0.5–0.7 s,  Wet ≥0.7 s

Usage:
  CUDA_VISIBLE_DEVICES=0 python generate_paper_tables.py \\
      --out_dir /home/rrame12/Desktop/Research/ASN/paper_output
"""

import os, sys, glob, json, argparse, warnings
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

# Perceptual metrics
from pesq import pesq as _pesq_fn, NoUtterancesError
from pystoi import stoi as _stoi_fn

sys.path.insert(0, os.path.dirname(__file__))

SR  = 16000
EPS = 1e-8
ASN_ROOT = os.path.join(os.path.dirname(__file__), "..")

# ─────────────────────────────────────────────────────────────────────────────
# Tier labels
# ─────────────────────────────────────────────────────────────────────────────

def snr_tier(v):
    if v < 0:   return "Hard"
    if v < 10:  return "Medium"
    return              "Easy"

def rt60_tier(v):
    if v < 0.5: return "Dry"
    if v < 0.7: return "Mild"
    return              "Wet"

def spk_label(v):
    return f"{v} spk" if v > 0 else "?"

# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def _rj(p):
    with open(p) as f: return json.load(f)

def _wj(p, o):
    with open(p, "w") as f: json.dump(o, f, indent=2)

def _safe_fname(s):
    """Replace characters that are invalid in filenames."""
    return s.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")

def load_v7(run_dir, ckpt, device):
    from train_graph_spectral_film_unet_v7 import GraphSpectralFiLMUNetV7, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphSpectralFiLMUNetV7(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft", 512)),
        hop=int(a.get("hop", 128)), base=int(a.get("base", 32)),
        depth=int(a.get("depth", 4)), drop=0.0,
        graph_dim=int(a.get("graph_dim", 128)),
        graph_heads=int(a.get("graph_heads", 4)),
        graph_dropout=0.0, use_graph=bool(a.get("graph_enabled", not a.get("disable_graph", False))),
        multiscale=bool(a.get("multiscale", False)))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

def load_v8(run_dir, ckpt, device):
    from train_graph_input_crossnode_v8 import GraphInputCrossNodeV8, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphInputCrossNodeV8(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft", 512)),
        hop=int(a.get("hop", 128)), base=int(a.get("base", 32)),
        depth=int(a.get("depth", 4)), drop=0.0,
        graph_dim=int(a.get("graph_dim", 128)),
        graph_heads=int(a.get("graph_heads", 4)),
        graph_dropout=0.0,
        use_graph=bool(a.get("graph_enabled", not a.get("disable_graph", False))))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

def load_v8_abl(run_dir, ckpt, device):
    from train_graph_ablations_v8 import GraphAblationV8, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphAblationV8(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft", 512)),
        hop=int(a.get("hop", 128)), base=int(a.get("base", 32)),
        depth=int(a.get("depth", 4)), drop=0.0,
        graph_dim=int(a.get("graph_dim", 128)),
        graph_heads=int(a.get("graph_heads", 4)),
        graph_dropout=0.0,
        use_graph=bool(a.get("graph_enabled", not a.get("disable_graph", False))),
        adj_mode=str(a.get("adj_mode", "fixed")),
        inject_level1=bool(a.get("inject_level1", False)))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

def load_v9(run_dir, ckpt, device):
    from train_graph_psm_v9 import GraphPSMV9, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphPSMV9(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft", 1024)),
        hop=int(a.get("hop", 256)), base=int(a.get("base", 32)),
        depth=int(a.get("depth", 4)), drop=0.0,
        graph_dim=int(a.get("graph_dim", 128)),
        graph_heads=int(a.get("graph_heads", 4)),
        graph_dropout=0.0,
        use_graph=bool(a.get("graph_enabled", not a.get("disable_graph", False))))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

def load_v9_all(run_dir, ckpt, device):
    from train_graph_alltoall_v9 import GraphAllToAllV9, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphAllToAllV9(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft", 1024)),
        hop=int(a.get("hop", 256)), base=int(a.get("base", 32)),
        depth=int(a.get("depth", 4)), drop=0.0,
        graph_dim=int(a.get("graph_dim", 128)),
        graph_heads=int(a.get("graph_heads", 4)),
        graph_dropout=0.0, use_graph=True)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

def load_v10(run_dir, ckpt, device):
    from train_graph_cirm_v10 import GraphCIRMV10, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphCIRMV10(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft", 1024)),
        hop=int(a.get("hop", 256)), base=int(a.get("base", 32)),
        depth=int(a.get("depth", 4)), drop=0.0,
        graph_dim=int(a.get("graph_dim", 128)),
        graph_heads=int(a.get("graph_heads", 4)),
        graph_dropout=0.0,
        use_graph=bool(a.get("graph_enabled", not a.get("disable_graph", False))),
        mask_scale=float(a.get("mask_scale", 10.0)))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

def load_v10_all(run_dir, ckpt, device):
    from train_graph_alltoall_v10 import GraphAllToAllV10, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphAllToAllV10(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft", 1024)),
        hop=int(a.get("hop", 256)), base=int(a.get("base", 32)),
        depth=int(a.get("depth", 4)), drop=0.0,
        graph_dim=int(a.get("graph_dim", 128)),
        graph_heads=int(a.get("graph_heads", 4)),
        graph_dropout=0.0, use_graph=True,
        mask_scale=float(a.get("mask_scale", 10.0)))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

# ─────────────────────────────────────────────────────────────────────────────
# Model registries
# ─────────────────────────────────────────────────────────────────────────────

# Table 1 models — (short_name, legacy_cache_name, run_dir, loader_fn)
TABLE1_MODELS = [
    ("Local (IRM)",      "V8-OFF",      f"{ASN_ROOT}/runs_input_crossnode_v8_off",  load_v8),
    ("Graph-F (IRM)",    "V8-ON",       f"{ASN_ROOT}/runs_input_crossnode_v8_on",   load_v8),
    ("Graph-All (IRM)",  "V8-alltoall", f"{ASN_ROOT}/runs_ablation_v8_alltoall",    load_v8_abl),
    ("Local (PSM)",      "V9-OFF",      f"{ASN_ROOT}/runs_psm_v9_off",              load_v9),
    ("Graph-F (PSM)",    "V9-ON",       f"{ASN_ROOT}/runs_psm_v9_on",               load_v9),
    ("Graph-All (PSM)",  None,          f"{ASN_ROOT}/runs_alltoall_v9_on",           load_v9_all),
    ("Local (cIRM)",     "V10-OFF",     f"{ASN_ROOT}/runs_cirm_v10_off",             load_v10),
    ("Graph-F (cIRM)",   "V10-ON",      f"{ASN_ROOT}/runs_cirm_v10_on",              load_v10),
    ("Graph-All (cIRM)", None,          f"{ASN_ROOT}/runs_alltoall_v10_on",          load_v10_all),
]

# Table 2 models
TABLE2_MODELS = [
    ("Local (IRM)",          "V8-OFF",     f"{ASN_ROOT}/runs_input_crossnode_v8_off",  load_v8),
    ("Bottleneck-only (V7)", "V7-OFF",     f"{ASN_ROOT}/runs_spectral_film_v7_off",    load_v7),
    ("Bottleneck+FiLM (V7)", "V7-ON",      f"{ASN_ROOT}/runs_spectral_film_v7_on",     load_v7),
    ("Graph-F (IRM / V8)",   "V8-ON",      f"{ASN_ROOT}/runs_input_crossnode_v8_on",   load_v8),
    ("Graph-All (IRM / V8)", "V8-alltoall",f"{ASN_ROOT}/runs_ablation_v8_alltoall",    load_v8_abl),
]

SPLITS = [
    ("Standard", "/home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk"),
    ("Hard",     "/home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_hard"),
    ("Worst",    "/home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_worst"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ShardMeta:
    def __init__(self, split_dir):
        self.split_dir = split_dir; self._c = {}
    def _load(self, sid):
        if sid in self._c: return self._c[sid]
        p = os.path.join(self.split_dir, f"shard_{sid:04d}.jsonl")
        if not os.path.exists(p): return []
        with open(p) as f: recs = [json.loads(l) for l in f if l.strip()]
        self._c[sid] = recs; return recs
    def get(self, sid, li):
        recs = self._load(sid)
        return recs[li] if recs and li < len(recs) else None

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split_dir):
        self.paths = sorted(glob.glob(os.path.join(split_dir, "shard_*.npz")))
        if not self.paths: raise RuntimeError(f"No shards in {split_dir}")
        self.index = []
        for sid, sp in enumerate(self.paths):
            with np.load(sp) as d:
                for i in range(d["Y"].shape[0]): self.index.append((sid, i))
        self._store = {}
    def __len__(self): return len(self.index)
    def _load(self, sid):
        if sid not in self._store:
            d = np.load(self.paths[sid])
            self._store[sid] = {k: d[k] for k in d.files}; d.close()
        return self._store[sid]
    def __getitem__(self, idx):
        sid, li = self.index[idx]; sh = self._load(sid)
        return {"Y": torch.from_numpy(sh["Y"][li]).float(),
                "S": torch.from_numpy(sh["target_refclean"][li]).float(),
                "A": torch.from_numpy(sh["adj"][li].astype(np.float32)).float(),
                "sid": sid, "li": li}

def collate(batch):
    return {"Y": torch.stack([b["Y"] for b in batch]),
            "S": torch.stack([b["S"] for b in batch]),
            "A": torch.stack([b["A"] for b in batch]),
            "sid": [b["sid"] for b in batch],
            "li":  [b["li"]  for b in batch]}

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _safe_pesq(ref, deg, sr=SR):
    """pesq wrapper: returns NaN on failure (silence, too short, etc.)."""
    try:
        mode = "wb" if sr >= 16000 else "nb"
        return float(_pesq_fn(sr, ref, deg, mode))
    except Exception:
        return float("nan")

def _safe_stoi(ref, deg, sr=SR):
    try:
        return float(_stoi_fn(ref, deg, sr, extended=False))
    except Exception:
        return float("nan")

def _norm(x):
    """Peak-normalise a 1-D numpy waveform to avoid PESQ clipping errors."""
    mx = np.abs(x).max()
    return x / (mx + 1e-9)

@torch.no_grad()
def evaluate(model, compute_si_sdr, loader, device, meta):
    rows = []
    for batch in tqdm(loader, desc="  eval", ncols=88, leave=False):
        Y = batch["Y"].to(device); S = batch["S"].to(device); A = batch["A"].to(device)
        Yhat    = model(Y, A)
        sdr_out = compute_si_sdr(Yhat,       S)
        sdr_in  = compute_si_sdr(Y[:,:,0,:], S)
        B, K, _ = Yhat.shape

        # Move to CPU numpy for perceptual metrics
        Yhat_np = Yhat.cpu().numpy()          # (B, K, T)
        Y_np    = Y.cpu().numpy()             # (B, K, M, T)  — use ch 0
        S_np    = S.cpu().numpy()             # (B, K, T)

        for b in range(B):
            m   = meta.get(batch["sid"][b], batch["li"][b])
            snr = float(m["snr_db"]) if m else float("nan")
            rt60= float(m["rt60"])   if m else float("nan")
            spt = m["speakers_per_table"] if m else [None]*K
            for k in range(K):
                lspk = spt[k] if k < len(spt) else None
                ref  = _norm(S_np[b, k])
                deg_in  = _norm(Y_np[b, k, 0])
                deg_out = _norm(Yhat_np[b, k])

                rows.append({
                    "sdr_in":   float(sdr_in[b,k].item()),
                    "sdr_out":  float(sdr_out[b,k].item()),
                    "delta":    float(sdr_out[b,k].item() - sdr_in[b,k].item()),
                    "pesq_in":  _safe_pesq(ref, deg_in),
                    "pesq_out": _safe_pesq(ref, deg_out),
                    "stoi_in":  _safe_stoi(ref, deg_in),
                    "stoi_out": _safe_stoi(ref, deg_out),
                    "snr": snr, "rt60": rt60,
                    "spk": int(lspk) if lspk is not None else -1,
                    "g_snr":  snr_tier(snr)    if m else "?",
                    "g_rt60": rt60_tier(rt60)  if m else "?",
                    "g_spk":  spk_label(int(lspk)) if lspk is not None else "?",
                })
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# Load or compute all data for a model list
# ─────────────────────────────────────────────────────────────────────────────

def _has_perceptual(rows):
    """True if cache already contains pesq_out / stoi_out fields."""
    return rows and "pesq_out" in rows[0] and "stoi_out" in rows[0]

def load_all(registry, splits_to_eval, cache_dir, device, batch_size, num_workers):
    """Returns data[split_name][model_name] = list of per-scene rows.

    Cache strategy:
      • New evaluations: saved to  <split>_<short_name>.json  (includes PESQ/STOI)
      • Legacy caches (SI-SDR only): loaded from old files, but model is
        re-evaluated to add PESQ/STOI → saved to new-style cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    data     = {s[0]: {} for s in splits_to_eval}
    # Keep loaders alive across models within a split
    loaders  = {}

    def get_loader(split_name, data_root):
        if split_name not in loaders:
            split_dir = os.path.join(data_root, "test")
            ds = Dataset(split_dir)
            loaders[split_name] = (
                split_dir,
                ShardMeta(split_dir),
                torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, collate_fn=collate,
                    pin_memory=True, drop_last=False,
                    persistent_workers=(num_workers > 0)))
        return loaders[split_name]

    for split_name, data_root in splits_to_eval:
        for short_name, legacy_name, run_dir, loader_fn in registry:
            new_cache  = os.path.join(cache_dir, f"{split_name}_{_safe_fname(short_name)}.json")
            legacy_cache = (os.path.join(cache_dir, f"{split_name}_{legacy_name}.json")
                            if legacy_name else None)

            # ── 1. New cache exists and has perceptual metrics ──────────────
            if os.path.exists(new_cache):
                rows = _rj(new_cache)["rows"]
                if _has_perceptual(rows):
                    print(f"[cached]         {split_name}/{short_name}")
                    data[split_name][short_name] = rows
                    continue
                # new cache exists but lacks PESQ/STOI → fall through to re-eval

            # ── 2. Legacy cache exists but lacks perceptual → re-eval ───────
            if legacy_cache and os.path.exists(legacy_cache):
                print(f"[re-eval PESQ]   {split_name}/{short_name}  (legacy cache has no PESQ/STOI)")
            else:
                print(f"[eval]           {split_name}/{short_name}")

            ckpt = os.path.join(run_dir, "best.pt")
            if not os.path.exists(ckpt):
                print(f"  [SKIP] no checkpoint at {ckpt}")
                # fall back to legacy cache without perceptual metrics
                if legacy_cache and os.path.exists(legacy_cache):
                    data[split_name][short_name] = _rj(legacy_cache)["rows"]
                continue

            _, meta, loader = get_loader(split_name, data_root)
            model, compute_si_sdr = loader_fn(run_dir, ckpt, device)
            rows = evaluate(model, compute_si_sdr, loader, device, meta)
            del model; torch.cuda.empty_cache()

            _wj(new_cache, {"split": split_name, "model": short_name, "rows": rows})
            data[split_name][short_name] = rows

    return data

# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nanmean(vals):
    v = [x for x in vals if x == x]  # filter NaN
    return sum(v) / len(v) if v else float("nan")

def agg(rows):
    if not rows: return None
    n = len(rows)
    return {
        "in":       sum(r["sdr_in"]  for r in rows) / n,
        "out":      sum(r["sdr_out"] for r in rows) / n,
        "delta":    sum(r["delta"]   for r in rows) / n,
        "pesq_in":  _nanmean([r.get("pesq_in",  float("nan")) for r in rows]),
        "pesq_out": _nanmean([r.get("pesq_out", float("nan")) for r in rows]),
        "stoi_in":  _nanmean([r.get("stoi_in",  float("nan")) for r in rows]),
        "stoi_out": _nanmean([r.get("stoi_out", float("nan")) for r in rows]),
        "n": n,
    }

def pool_splits(data, model_name, split_names):
    """Combine rows across multiple splits for breakdown."""
    combined = []
    for s in split_names:
        combined += data[s].get(model_name, [])
    return combined

def breakdown_agg(rows, key):
    buckets = defaultdict(list)
    for r in rows: buckets[r[key]].append(r)
    return {g: agg(v) for g, v in sorted(buckets.items())}

# ─────────────────────────────────────────────────────────────────────────────
# Format helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v, bold=False, color=None):
    """Format a float to 2 decimal places; optionally bold."""
    if v is None: return "--"
    s = f"{v:.2f}"
    if bold: s = r"\textbf{" + s + "}"
    return s

def fmt_delta(v, bold=False):
    if v is None: return "--"
    s = f"{v:+.2f}"
    if bold: s = r"\textbf{" + s + "}"
    return f"${s}$"

# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Main results
# ─────────────────────────────────────────────────────────────────────────────

def _build_row_defs(data):
    """Shared row definitions: (group, condition_label, rows_fn)."""
    split_names = [s[0] for s in SPLITS]

    def gbnum(snr_fn=None, rt60_fn=None, spk_val=None, splits=None):
        _splits = splits or split_names
        def fn(m):
            rows = pool_splits(data, m, _splits)
            return [r for r in rows if
                    (snr_fn  is None or snr_fn(r["snr"]))   and
                    (rt60_fn is None or rt60_fn(r["rt60"])) and
                    (spk_val is None or r["spk"] == spk_val)]
        return fn

    return [
        ("Overall",       "Standard",
         lambda m: data["Standard"].get(m, [])),
        ("",              "Hard",
         lambda m: data["Hard"].get(m, [])),
        ("",              "Worst",
         lambda m: data["Worst"].get(m, [])),
        ("Speaker count", "1 speaker",
         gbnum(spk_val=1, splits=["Standard", "Hard"])),
        ("",              "2 speakers",
         gbnum(spk_val=2, splits=["Standard", "Hard"])),
        ("",              "3 speakers",
         gbnum(spk_val=3, splits=split_names)),
        (r"SNR",          r"Easy ($\geq$10~dB)",
         gbnum(snr_fn=lambda v: v >= 10)),
        ("",              r"Medium (0--10~dB)",
         gbnum(snr_fn=lambda v: 0 <= v < 10)),
        ("",              r"Hard ($<$0~dB)",
         gbnum(snr_fn=lambda v: v < 0)),
        (r"RT60",         r"Dry ($<$0.5~s)",
         gbnum(rt60_fn=lambda v: v < 0.5)),
        ("",              r"Mild (0.5--0.7~s)",
         gbnum(rt60_fn=lambda v: 0.5 <= v < 0.7)),
        ("",              r"Wet ($\geq$0.7~s)",
         gbnum(rt60_fn=lambda v: v >= 0.7)),
    ]


def _single_metric_table(table_rows, model_names, metric_out, mix_in,
                          caption, label, fmt_fn, path):
    """Emit one table*: Mixture col + 9 model cols, grouped IRM/PSM/cIRM."""
    col_spec = "ll c " + " ".join(["c"] * len(model_names))
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering\setlength{\tabcolsep}{4pt}")
    L.append(r"\caption{" + caption + "}")
    L.append(r"\label{" + label + "}")
    L.append(r"\begin{tabular}{" + col_spec + "}")
    L.append(r"\toprule")
    L.append(r"\textbf{Setting} & \textbf{Condition} & \textbf{Mixture}"
             r" & \multicolumn{3}{c}{\textbf{IRM}}"
             r" & \multicolumn{3}{c}{\textbf{PSM}}"
             r" & \multicolumn{3}{c}{\textbf{cIRM}} \\")
    L.append(r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}")
    L.append(r"& & & Local & G-F & G-All & Local & G-F & G-All & Local & G-F & G-All \\")
    L.append(r"\midrule")

    prev_group = None
    for group, condition, rows_fn in table_rows:
        if prev_group is not None and group != "" and group != prev_group:
            L.append(r"\midrule")
        if group != "": prev_group = group

        mix_val = None
        for mn in model_names:
            a = agg(rows_fn(mn))
            if a:
                v = a.get(mix_in)
                if v is not None and v == v:  # not NaN
                    mix_val = v; break

        out_vals = []
        for mn in model_names:
            a = agg(rows_fn(mn))
            v = a.get(metric_out) if a else None
            out_vals.append(v if (v is not None and v == v) else None)

        valid = [v for v in out_vals if v is not None]
        best  = max(valid) if valid else None

        mix_str   = fmt_fn(mix_val)
        cell_strs = [fmt_fn(v, bold=(v is not None and best is not None
                                     and abs(v - best) < 0.005))
                     for v in out_vals]
        L.append((group if group else "") + " & " + condition
                 + " & " + mix_str + " & " + " & ".join(cell_strs) + r" \\")

    L.append(r"\bottomrule\end{tabular}\end{table*}")
    with open(path, "w") as f: f.write("\n".join(L) + "\n")
    print(f"Saved {path}")


def make_table1(data, out_dir):
    """Write SI-SDR, PESQ, and STOI tables (same row layout, different metric)."""
    model_names = [m[0] for m in TABLE1_MODELS]
    table_rows  = _build_row_defs(data)

    def fmt2(v, bold=False):
        if v is None: return "--"
        s = f"{v:.2f}"
        return r"\textbf{" + s + "}" if bold else s

    def fmt3(v, bold=False):  # 3 decimal places for STOI
        if v is None: return "--"
        s = f"{v:.3f}"
        return r"\textbf{" + s + "}" if bold else s

    # ── SI-SDR ────────────────────────────────────────────────────────────────
    _single_metric_table(
        table_rows, model_names, "out", "in",
        caption=(r"Mean output SI-SDR (dB). \emph{Mixture} = mean input SI-SDR. "
                 r"SNR/RT60 rows pool all three splits by tier; "
                 r"speaker rows use Standard + Hard. Best per row \textbf{bold}."),
        label="tab:results_sisdr",
        fmt_fn=fmt2,
        path=os.path.join(out_dir, "table_results_sisdr.tex"),
    )

    # ── PESQ ─────────────────────────────────────────────────────────────────
    _single_metric_table(
        table_rows, model_names, "pesq_out", "pesq_in",
        caption=(r"Mean output PESQ (WB-MOS, 1--4.5, higher is better). "
                 r"\emph{Mixture} = input PESQ. Best per row \textbf{bold}."),
        label="tab:results_pesq",
        fmt_fn=fmt2,
        path=os.path.join(out_dir, "table_results_pesq.tex"),
    )

    # ── STOI ─────────────────────────────────────────────────────────────────
    _single_metric_table(
        table_rows, model_names, "stoi_out", "stoi_in",
        caption=(r"Mean output STOI (0--1, higher is better). "
                 r"\emph{Mixture} = input STOI. Best per row \textbf{bold}."),
        label="tab:results_stoi",
        fmt_fn=fmt3,
        path=os.path.join(out_dir, "table_results_stoi.tex"),
    )

    # ── Also keep the original combined SI-SDR table (for backward compat) ──
    _single_metric_table(
        table_rows, model_names, "out", "in",
        caption=(r"Mean output SI-SDR (dB) across all models and conditions."),
        label="tab:main_results",
        fmt_fn=fmt2,
        path=os.path.join(out_dir, "table_results_main.tex"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Table 2: Architecture ablation
# ─────────────────────────────────────────────────────────────────────────────

def make_table2(data, out_dir):
    """
    Small ablation table: input injection OFF/ON × bottleneck graph OFF/ON.
    Shows V7 (bottleneck-only) vs V8 (input+bottleneck) vs Local.
    Columns: Standard SI-SDR out | Worst SI-SDR out | Graph gain (Worst)
    """
    model_names   = [m[0] for m in TABLE2_MODELS]
    local_name    = "Local (IRM)"

    # Architecture annotations
    arch_map = {
        "Local (IRM)":          (r"--",                r"--"),
        "Bottleneck-only (V7)": (r"--",                r"\checkmark"),
        "Bottleneck+FiLM (V7)": (r"--",                r"\checkmark~(multi)"),
        "Graph-F (IRM / V8)":   (r"\checkmark",        r"\checkmark"),
        "Graph-All (IRM / V8)": (r"\checkmark~(all-to-all)", r"\checkmark"),
    }

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(
        r"\caption{Architecture ablation: contribution of input-level neighbor injection "
        r"and bottleneck graph conditioning (IRM mask, n\_fft=512). "
        r"Output SI-SDR (dB) on Standard and Worst test splits. "
        r"$\Delta_\text{W}$ = graph gain on Worst split vs.\ Local baseline. "
        r"Best value per column in \textbf{bold}.}"
    )
    lines.append(r"\label{tab:arch_ablation}")
    lines.append(r"\begin{tabular}{lcc ccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{Model} & \multicolumn{2}{c}{Graph mechanism} "
        r"& \multicolumn{2}{c}{Output SI-SDR (dB)} & \multirow{2}{*}{$\Delta_\text{W}$} \\"
    )
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(
        r"& Input inj. & Bottleneck "
        r"& Standard & Worst & \\"
    )
    lines.append(r"\midrule")

    # Compute local baseline for graph gain
    local_rows_std   = data["Standard"].get(local_name, [])
    local_rows_worst = data["Worst"].get(local_name, [])
    local_std  = agg(local_rows_std)["out"]  if agg(local_rows_std)  else None
    local_worst= agg(local_rows_worst)["out"] if agg(local_rows_worst) else None

    # Collect all values for bolding
    std_vals   = []
    worst_vals = []
    delta_vals = []
    for mn in model_names:
        a_s = agg(data["Standard"].get(mn, []))
        a_w = agg(data["Worst"].get(mn, []))
        std_vals.append(a_s["out"]   if a_s  else None)
        worst_vals.append(a_w["out"] if a_w  else None)
        delta_vals.append(
            a_w["out"] - local_worst
            if (a_w and local_worst is not None and mn != local_name)
            else None
        )

    best_std   = max(v for v in std_vals   if v is not None)
    best_worst = max(v for v in worst_vals if v is not None)
    best_delta = max(v for v in delta_vals if v is not None)

    for i, mn in enumerate(model_names):
        inp_inj, bott = arch_map.get(mn, ("?", "?"))
        s_val = std_vals[i];   w_val = worst_vals[i];   d_val = delta_vals[i]

        s_str = fmt(s_val,  bold=(s_val  is not None and abs(s_val  - best_std)   < 0.005))
        w_str = fmt(w_val,  bold=(w_val  is not None and abs(w_val  - best_worst) < 0.005))
        d_str = (fmt_delta(d_val, bold=(d_val is not None and abs(d_val - best_delta) < 0.005))
                 if mn != local_name else "--")

        # Add midrule before V8 models
        if mn == "Graph-F (IRM / V8)":
            lines.append(r"\midrule")

        lines.append(
            f"{mn} & {inp_inj} & {bott} & {s_str} & {w_str} & {d_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = os.path.join(out_dir, "table_arch_ablation.tex")
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Print summary to stdout (also useful for quick sanity check)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(data1, data2):
    print("\n" + "="*90)
    print("TABLE 1 SUMMARY — Worst split")
    print("="*90)
    header = "%-22s %8s %7s %7s %7s %7s %7s" % (
        "Model", "SDR-in", "SDR-out", "ΔSDR", "PESQ-in", "PESQ-out", "STOI-out")
    print(header); print("-"*len(header))
    for mn, *_ in TABLE1_MODELS:
        a = agg(data1["Worst"].get(mn, []))
        if not a: print(f"  {mn:<20}  N/A"); continue
        print("  %-20s %8.2f %7.2f %7.2f %7.2f %8.2f %8.3f" % (
            mn, a["in"], a["out"], a["delta"],
            a["pesq_in"] if a["pesq_in"]==a["pesq_in"] else float("nan"),
            a["pesq_out"] if a["pesq_out"]==a["pesq_out"] else float("nan"),
            a["stoi_out"] if a["stoi_out"]==a["stoi_out"] else float("nan"),
        ))

    print("\n" + "="*72)
    print("TABLE 2 — Architecture ablation")
    print("="*72)
    header = f"{'Model':<28} {'Std out':>8} {'Worst out':>9} {'Gain(W)':>8}"
    print(header); print("-"*len(header))
    local_w = agg(data2["Worst"].get("Local (IRM)", []))
    local_w_out = local_w["out"] if local_w else 0.0
    for mn, *_ in TABLE2_MODELS:
        s = agg(data2["Standard"].get(mn, []))
        w = agg(data2["Worst"].get(mn,    []))
        sv = f"{s['out']:.2f}" if s else "  N/A"
        wv = f"{w['out']:.2f}" if w else "  N/A"
        gain = f"{w['out'] - local_w_out:+.2f}" if (w and mn != "Local (IRM)") else "    —"
        print(f"  {mn:<26} {sv:>8} {wv:>9} {gain:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",     default="/home/rrame12/Desktop/Research/ASN/paper_output")
    ap.add_argument("--batch",       type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpu",         default="0")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = os.path.join(args.out_dir, "per_scene")
    table_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    print(f"Device: {device}")

    # ── Evaluate Table 1 models ───────────────────────────────────────────────
    print("\n=== Loading Table 1 data ===")
    data1 = load_all(TABLE1_MODELS, SPLITS, cache_dir, device, args.batch, args.num_workers)

    # ── Evaluate Table 2 models (V7 needs eval; V8 reuses cache) ─────────────
    print("\n=== Loading Table 2 data (V7 architecture ablation) ===")
    data2 = load_all(TABLE2_MODELS, SPLITS, cache_dir, device, args.batch, args.num_workers)

    # ── Generate tables ───────────────────────────────────────────────────────
    print("\n=== Generating LaTeX tables ===")
    make_table1(data1, table_dir)
    make_table2(data2, table_dir)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(data1, data2)

    print(f"\nTables saved to: {table_dir}/")
    print("  table_results_main.tex   — Table 1 (comprehensive)")
    print("  table_arch_ablation.tex  — Table 2 (V7 vs V8 ablation)")


if __name__ == "__main__":
    main()
