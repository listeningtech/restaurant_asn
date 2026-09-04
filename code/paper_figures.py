#!/usr/bin/env python3
"""
paper_figures.py

Generate all paper figures and LaTeX tables from trained models.

Outputs (saved to --out_dir):
  figures/fig1_graph_gain_progression.pdf   Bar chart: graph gain vs condition severity
  figures/fig2_boxplots_delta.pdf           Box plots: per-scene SI-SDR delta by model × split
  figures/fig3_breakdown_spk.pdf            Grouped bars: delta by speaker count
  figures/fig4_breakdown_snr.pdf            Grouped bars: delta by SNR tier
  figures/fig5_breakdown_rt60.pdf           Grouped bars: delta by RT60 tier
  figures/fig6_combined_breakdown.pdf       3-panel breakdown (spk/snr/rt60) for paper
  tables/table_overall.tex                  LaTeX: main results table
  tables/table_breakdown.tex               LaTeX: breakdown table
  per_scene/                               Per-scene CSV/JSON for each model × split

Usage:
  CUDA_VISIBLE_DEVICES=0 python paper_figures.py \
    --out_dir /home/rrame12/Desktop/Research/ASN/paper_output
"""

import os, sys, glob, json, argparse, csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

SR  = 16000
EPS = 1e-8
ASN_ROOT = os.path.join(os.path.dirname(__file__), "..")

# ─────────────────────────────────────────────────────────────────────────────
# Paper style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# Colors per model group
COLORS = {
    "OFF":      "#888888",
    "ON_fixed": "#2196F3",
    "ON_ext":   "#64B5F6",
    "ON_all":   "#E53935",
    "ON_l1":    "#90CAF9",
    "V9_OFF":   "#BDBDBD",
    "V9_ON":    "#FF9800",
    "V9_all":   "#E65100",
    "V10_OFF":  "#CFCFCF",
    "V10_ON":   "#4CAF50",
    "V10_all":  "#1B5E20",
}

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

def load_v8(run_dir, ckpt, device):
    from train_graph_input_crossnode_v8 import GraphInputCrossNodeV8, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphInputCrossNodeV8(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft",512)),
        hop=int(a.get("hop",128)), base=int(a.get("base",32)),
        depth=int(a.get("depth",4)), drop=0.0,
        graph_dim=int(a.get("graph_dim",128)), graph_heads=int(a.get("graph_heads",4)),
        graph_dropout=0.0, use_graph=bool(a.get("graph_enabled", not a.get("disable_graph",False))))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

def load_v8_abl(run_dir, ckpt, device):
    from train_graph_ablations_v8 import GraphAblationV8, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphAblationV8(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft",512)),
        hop=int(a.get("hop",128)), base=int(a.get("base",32)),
        depth=int(a.get("depth",4)), drop=0.0,
        graph_dim=int(a.get("graph_dim",128)), graph_heads=int(a.get("graph_heads",4)),
        graph_dropout=0.0, use_graph=bool(a.get("graph_enabled", not a.get("disable_graph",False))),
        adj_mode=str(a.get("adj_mode","fixed")), inject_level1=bool(a.get("inject_level1",False)))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

def load_v9(run_dir, ckpt, device):
    from train_graph_psm_v9 import GraphPSMV9, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphPSMV9(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft",1024)),
        hop=int(a.get("hop",256)), base=int(a.get("base",32)),
        depth=int(a.get("depth",4)), drop=0.0,
        graph_dim=int(a.get("graph_dim",128)), graph_heads=int(a.get("graph_heads",4)),
        graph_dropout=0.0, use_graph=bool(a.get("graph_enabled", not a.get("disable_graph",False))))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

def load_v9_all(run_dir, ckpt, device):
    from train_graph_alltoall_v9 import GraphAllToAllV9, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphAllToAllV9(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft",1024)),
        hop=int(a.get("hop",256)), base=int(a.get("base",32)),
        depth=int(a.get("depth",4)), drop=0.0,
        graph_dim=int(a.get("graph_dim",128)), graph_heads=int(a.get("graph_heads",4)),
        graph_dropout=0.0, use_graph=True)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

def load_v10(run_dir, ckpt, device):
    from train_graph_cirm_v10 import GraphCIRMV10, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphCIRMV10(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft",1024)),
        hop=int(a.get("hop",256)), base=int(a.get("base",32)),
        depth=int(a.get("depth",4)), drop=0.0,
        graph_dim=int(a.get("graph_dim",128)), graph_heads=int(a.get("graph_heads",4)),
        graph_dropout=0.0, use_graph=bool(a.get("graph_enabled", not a.get("disable_graph",False))),
        mask_scale=float(a.get("mask_scale",10.0)))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

def load_v10_all(run_dir, ckpt, device):
    from train_graph_alltoall_v10 import GraphAllToAllV10, compute_si_sdr
    a = _rj(os.path.join(run_dir, "args.json"))
    m = GraphAllToAllV10(
        n_mics=int(a["resolved_n_mics"]), n_fft=int(a.get("nfft",1024)),
        hop=int(a.get("hop",256)), base=int(a.get("base",32)),
        depth=int(a.get("depth",4)), drop=0.0,
        graph_dim=int(a.get("graph_dim",128)), graph_heads=int(a.get("graph_heads",4)),
        graph_dropout=0.0, use_graph=True,
        mask_scale=float(a.get("mask_scale",10.0)))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m.to(device).eval(), compute_si_sdr

# (short_name, graph, mask, adj, color_key, run_dir, loader_fn)
# Naming convention: Local (MASK), Graph-F (MASK), Graph-All (MASK)
#   MASK ∈ {IRM, PSM, cIRM}
#   Graph-F  = fixed adjacency  (topology from dataset)
#   Graph-All = all-to-all adjacency (fully connected override)
MODEL_REGISTRY = [
    ("Local (IRM)",      "OFF", "IRM",  "—",        "OFF",      f"{ASN_ROOT}/runs_input_crossnode_v8_off",    load_v8),
    ("Graph-F (IRM)",    "ON",  "IRM",  "fixed",    "ON_fixed", f"{ASN_ROOT}/runs_input_crossnode_v8_on",     load_v8),
    ("Graph-F+ (IRM)",   "ON",  "IRM",  "fixed",    "ON_ext",   f"{ASN_ROOT}/runs_input_crossnode_v8_on_ext", load_v8),
    ("Graph-All (IRM)",  "ON",  "IRM",  "alltoall", "ON_all",   f"{ASN_ROOT}/runs_ablation_v8_alltoall",      load_v8_abl),
    ("Graph-L1 (IRM)",   "ON",  "IRM",  "fixed+L1", "ON_l1",    f"{ASN_ROOT}/runs_ablation_v8_multilevel",    load_v8_abl),
    ("Local (PSM)",      "OFF", "PSM",  "—",        "V9_OFF",   f"{ASN_ROOT}/runs_psm_v9_off",                load_v9),
    ("Graph-F (PSM)",    "ON",  "PSM",  "fixed",    "V9_ON",    f"{ASN_ROOT}/runs_psm_v9_on",                 load_v9),
    ("Graph-All (PSM)",  "ON",  "PSM",  "alltoall", "V9_all",   f"{ASN_ROOT}/runs_alltoall_v9_on",            load_v9_all),
    ("Local (cIRM)",     "OFF", "cIRM", "—",        "V10_OFF",  f"{ASN_ROOT}/runs_cirm_v10_off",              load_v10),
    ("Graph-F (cIRM)",   "ON",  "cIRM", "fixed",    "V10_ON",   f"{ASN_ROOT}/runs_cirm_v10_on",               load_v10),
    ("Graph-All (cIRM)", "ON",  "cIRM", "alltoall", "V10_all",  f"{ASN_ROOT}/runs_alltoall_v10_on",           load_v10_all),
]

# Map new paper names → old cache filenames (to reuse existing evaluations)
LEGACY_CACHE_NAMES = {
    "Local (IRM)":      "V8-OFF",
    "Graph-F (IRM)":    "V8-ON",
    "Graph-F+ (IRM)":   "V8-ON-ext",
    "Graph-All (IRM)":  "V8-alltoall",
    "Graph-L1 (IRM)":   "V8-L1inj",
    "Local (PSM)":      "V9-OFF",
    "Graph-F (PSM)":    "V9-ON",
    "Local (cIRM)":     "V10-OFF",
    "Graph-F (cIRM)":   "V10-ON",
}

SPLITS = [
    ("Standard", "/home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk",  "SNR 5–20 dB\nRT60 0.3–0.6s"),
    ("Hard",     "/home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_hard",     "SNR 0–8 dB\nRT60 0.5–0.8s"),
    ("Worst",    "/home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_worst",    "SNR −8–0 dB\nRT60 0.6–0.9s"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rj(p):
    with open(p) as f: return json.load(f)

def _wj(p, o):
    with open(p,"w") as f: json.dump(o, f, indent=2)

def _ensure(p): os.makedirs(p, exist_ok=True)

def snr_tier(v):
    if v <  0: return "SNR<0"
    if v <  5: return "0≤SNR<5"
    if v < 10: return "5≤SNR<10"
    return            "SNR≥10"

def rt60_tier(v):
    if v < 0.5:  return "RT60<0.5"
    if v < 0.65: return "0.5≤RT60<0.65"
    if v < 0.80: return "0.65≤RT60<0.8"
    return              "RT60≥0.8"


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
# Evaluation — returns per-scene rows
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, compute_si_sdr, loader, device, meta):
    rows = []
    for batch in tqdm(loader, desc="  eval", ncols=90, leave=False):
        Y = batch["Y"].to(device); S = batch["S"].to(device); A = batch["A"].to(device)
        Yhat    = model(Y, A)
        sdr_out = compute_si_sdr(Yhat,       S)
        sdr_in  = compute_si_sdr(Y[:,:,0,:], S)
        B, K, _ = Yhat.shape
        for b in range(B):
            m   = meta.get(batch["sid"][b], batch["li"][b])
            snr = float(m["snr_db"]) if m else float("nan")
            rt60= float(m["rt60"])   if m else float("nan")
            spt = m["speakers_per_table"] if m else [None]*K
            for k in range(K):
                lspk = spt[k] if k < len(spt) else None
                rows.append({
                    "sdr_in":  float(sdr_in[b,k].item()),
                    "sdr_out": float(sdr_out[b,k].item()),
                    "delta":   float(sdr_out[b,k].item() - sdr_in[b,k].item()),
                    "snr":     snr, "rt60": rt60,
                    "spk":     int(lspk) if lspk is not None else -1,
                    "g_snr":   snr_tier(snr)  if m else "?",
                    "g_rt60":  rt60_tier(rt60) if m else "?",
                    "g_spk":   f"{lspk}spk"   if lspk is not None else "?",
                })
    return rows

def agg(rows):
    if not rows: return {"in": float("nan"), "out": float("nan"), "delta": float("nan"), "n": 0}
    n = len(rows)
    return {"in":    sum(r["sdr_in"]  for r in rows)/n,
            "out":   sum(r["sdr_out"] for r in rows)/n,
            "delta": sum(r["delta"]   for r in rows)/n, "n": n}

def breakdown(rows, key):
    buckets = defaultdict(list)
    for r in rows: buckets[r[key]].append(r)
    return {g: agg(v) for g, v in sorted(buckets.items())}


# ─────────────────────────────────────────────────────────────────────────────
# Load or compute all per-scene data
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data(out_dir, device, batch_size, num_workers):
    """
    Returns nested dict: data[split_name][model_name] = list of per-scene rows
    Uses per_scene/ cache directory to avoid re-evaluation.
    """
    cache_dir = os.path.join(out_dir, "per_scene")
    _ensure(cache_dir)

    data = {}
    for split_name, data_root, _ in SPLITS:
        data[split_name] = {}
        split_dir = os.path.join(data_root, "test")
        meta      = ShardMeta(split_dir)
        ds        = Dataset(split_dir)
        loader    = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate,
            pin_memory=True, drop_last=False,
            persistent_workers=(num_workers > 0))

        for short_name, graph, mask, adj, color_key, run_dir, loader_fn in MODEL_REGISTRY:
            cache_path = os.path.join(cache_dir, f"{split_name}_{short_name}.json")
            # Also check legacy cache under old model name
            legacy_name = LEGACY_CACHE_NAMES.get(short_name)
            legacy_path = os.path.join(cache_dir, f"{split_name}_{legacy_name}.json") \
                          if legacy_name else None
            if os.path.exists(cache_path):
                print(f"[cached] {split_name} / {short_name}")
                data[split_name][short_name] = _rj(cache_path)["rows"]
                continue
            if legacy_path and os.path.exists(legacy_path):
                print(f"[cached-legacy] {split_name} / {short_name}  (from {legacy_name})")
                data[split_name][short_name] = _rj(legacy_path)["rows"]
                continue

            ckpt = os.path.join(run_dir, "best.pt")
            if not os.path.exists(ckpt):
                print(f"[SKIP]   {split_name} / {short_name} — no checkpoint")
                continue

            print(f"[eval]   {split_name} / {short_name}")
            model, compute_si_sdr = loader_fn(run_dir, ckpt, device)
            rows = evaluate(model, compute_si_sdr, loader, device, meta)
            del model; torch.cuda.empty_cache()

            _wj(cache_path, {"split": split_name, "model": short_name, "rows": rows})
            data[split_name][short_name] = rows

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Graph gain progression (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def fig_graph_gain_progression(data, out_dir):
    # ON models and their matching OFF baseline — one entry per (mask, adj) combination
    pairs = [
        ("Graph-F (IRM)",    "Local (IRM)",   "Graph-F (IRM)",    COLORS["ON_fixed"]),
        ("Graph-All (IRM)",  "Local (IRM)",   "Graph-All (IRM)",  COLORS["ON_all"]),
        ("Graph-F (PSM)",    "Local (PSM)",   "Graph-F (PSM)",    COLORS["V9_ON"]),
        ("Graph-All (PSM)",  "Local (PSM)",   "Graph-All (PSM)",  COLORS["V9_all"]),
        ("Graph-F (cIRM)",   "Local (cIRM)",  "Graph-F (cIRM)",   COLORS["V10_ON"]),
        ("Graph-All (cIRM)", "Local (cIRM)",  "Graph-All (cIRM)", COLORS["V10_all"]),
    ]
    split_names  = [s[0] for s in SPLITS]
    split_labels = ["Standard\n(SNR 5–20dB)", "Hard\n(SNR 0–8dB)", "Worst\n(SNR −8–0dB)"]

    x      = np.arange(len(split_names))
    width  = 0.13
    offsets = np.linspace(-(len(pairs)-1)*width/2, (len(pairs)-1)*width/2, len(pairs))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, (on_name, off_name, label, color) in enumerate(pairs):
        gains = []
        for split_name in split_names:
            on_rows  = data[split_name].get(on_name,  [])
            off_rows = data[split_name].get(off_name, [])
            g = agg(on_rows)["out"] - agg(off_rows)["out"] if (on_rows and off_rows) else 0.0
            gains.append(g)
        bars = ax.bar(x + offsets[i], gains, width, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, gains):
            if val > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.06,
                        f"{val:+.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xlabel("Test Condition")
    ax.set_ylabel("ΔSI-SDR: Graph − Local (dB)")
    ax.set_title("Graph Gain vs Acoustic Difficulty\n(ΔSI-SDR: ON – OFF)", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(split_labels)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 7.0)
    ax.legend(loc="upper left", ncol=2, framealpha=0.9, fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig1_graph_gain_progression.pdf")
    fig.savefig(p); fig.savefig(p.replace(".pdf",".png")); plt.close(fig)
    print(f"Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Box plots of per-scene delta SI-SDR
# ─────────────────────────────────────────────────────────────────────────────

def fig_boxplots(data, out_dir):
    # Key models: one local + all graph variants per mask type
    key_models = [
        "Local (IRM)", "Graph-F (IRM)", "Graph-All (IRM)",
        "Graph-F (PSM)", "Graph-All (PSM)",
        "Graph-F (cIRM)", "Graph-All (cIRM)",
    ]
    key_colors = [
        COLORS["OFF"], COLORS["ON_fixed"], COLORS["ON_all"],
        COLORS["V9_ON"], COLORS["V9_all"],
        COLORS["V10_ON"], COLORS["V10_all"],
    ]
    split_names  = [s[0] for s in SPLITS]
    split_labels = ["Standard (SNR 5–20dB)", "Hard (SNR 0–8dB)", "Worst (SNR −8–0dB)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for col, (split_name, ax) in enumerate(zip(split_names, axes)):
        plot_data  = []
        positions  = []
        tick_pos   = []
        tick_label = []
        colors_used = []
        pos = 1
        for m_name, color in zip(key_models, key_colors):
            rows = data[split_name].get(m_name, [])
            if rows:
                deltas = [r["delta"] for r in rows]
                plot_data.append(deltas)
                positions.append(pos)
                tick_pos.append(pos)
                # Shorter tick labels
                lbl = m_name.replace("Graph-All", "G-All").replace("Graph-F", "G-F").replace("Local", "Local")
                tick_label.append(lbl)
                colors_used.append(color)
                pos += 1

        bp = ax.boxplot(plot_data, positions=positions, patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(marker="o", markersize=2, alpha=0.3),
                        medianprops=dict(color="black", linewidth=2.5),
                        whiskerprops=dict(linewidth=1.0),
                        capprops=dict(linewidth=1.0))
        for patch, color in zip(bp["boxes"], colors_used):
            patch.set_facecolor(color); patch.set_alpha(0.75)

        # Title only shows the condition — no model info in subplot title
        ax.set_title(split_labels[col], fontsize=10)
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_label, fontsize=7.5, rotation=30, ha="right")
        ax.set_ylabel("ΔSI-SDR (dB)" if col == 0 else "")
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")

    fig.suptitle("Per-Scene ΔSI-SDR Distribution by Model and Test Condition", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig2_boxplots_delta.pdf")
    fig.savefig(p); fig.savefig(p.replace(".pdf",".png")); plt.close(fig)
    print(f"Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Combined 3-panel breakdown (spk / SNR / RT60) for best models
# ─────────────────────────────────────────────────────────────────────────────

def fig_combined_breakdown(data, out_dir):
    # Use worst-case split; 2-panel breakdown (SPK + SNR) for conciseness
    split_name = "Worst"
    key_models  = [
        "Local (IRM)", "Graph-F (IRM)", "Graph-All (IRM)",
        "Graph-F (PSM)", "Graph-All (PSM)",
        "Graph-F (cIRM)", "Graph-All (cIRM)",
    ]
    key_colors  = [
        COLORS["OFF"], COLORS["ON_fixed"], COLORS["ON_all"],
        COLORS["V9_ON"], COLORS["V9_all"],
        COLORS["V10_ON"], COLORS["V10_all"],
    ]
    key_labels  = [
        "Local (IRM)", "G-F (IRM)", "G-All (IRM)",
        "G-F (PSM)", "G-All (PSM)",
        "G-F (cIRM)", "G-All (cIRM)",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    breakdown_configs = [
        ("g_spk",  "Speaker Count",  axes[0]),
        ("g_snr",  "SNR Tier",       axes[1]),
    ]

    for key, title, ax in breakdown_configs:
        all_groups = sorted({r[key] for m in key_models
                             for r in data[split_name].get(m, [])})
        x       = np.arange(len(all_groups))
        n       = len(key_models)
        width   = 0.10
        offsets = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)
        for i, (m_name, color, label) in enumerate(zip(key_models, key_colors, key_labels)):
            rows = data[split_name].get(m_name, [])
            bd   = breakdown(rows, key)
            vals = [bd.get(g, {}).get("delta", 0.0) for g in all_groups]
            ax.bar(x + offsets[i], vals, width, label=label, color=color,
                   edgecolor="white", linewidth=0.4, alpha=0.85)

        short_groups = [g.replace("≤","-").replace("≥","≥").replace("spk"," spk") for g in all_groups]
        ax.set_xticks(x); ax.set_xticklabels(short_groups, fontsize=8.5, rotation=10, ha="right")
        ax.set_title(title)
        ax.set_ylabel("ΔSI-SDR (dB)")
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        if title == "Speaker Count":
            ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    fig.suptitle("ΔSI-SDR Breakdown — Worst-Case Split (SNR −8–0 dB, RT60 0.6–0.9s)",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig3_breakdown_worst.pdf")
    fig.savefig(p); fig.savefig(p.replace(".pdf",".png")); plt.close(fig)
    print(f"Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Graph gain heatmap across all splits × models
# ─────────────────────────────────────────────────────────────────────────────

def fig_gain_heatmap(data, out_dir):
    on_off_pairs = [
        ("Graph-F (IRM)",    "Local (IRM)",  "Graph-F\n(IRM, fixed)"),
        ("Graph-F+ (IRM)",   "Local (IRM)",  "Graph-F+\n(IRM, fixed)"),
        ("Graph-All (IRM)",  "Local (IRM)",  "Graph-All\n(IRM)"),
        ("Graph-L1 (IRM)",   "Local (IRM)",  "Graph-L1\n(IRM)"),
        ("Graph-F (PSM)",    "Local (PSM)",  "Graph-F\n(PSM)"),
        ("Graph-All (PSM)",  "Local (PSM)",  "Graph-All\n(PSM)"),
        ("Graph-F (cIRM)",   "Local (cIRM)", "Graph-F\n(cIRM)"),
        ("Graph-All (cIRM)", "Local (cIRM)", "Graph-All\n(cIRM)"),
    ]
    split_names  = [s[0] for s in SPLITS]
    split_labels = ["Standard\n(SNR 5–20)", "Hard\n(SNR 0–8)", "Worst\n(SNR −8–0)"]
    model_labels = [p[2] for p in on_off_pairs]

    matrix = np.zeros((len(on_off_pairs), len(split_names)))
    for j, split_name in enumerate(split_names):
        for i, (on_n, off_n, _) in enumerate(on_off_pairs):
            on_r  = data[split_name].get(on_n,  [])
            off_r = data[split_name].get(off_n, [])
            if on_r and off_r:
                matrix[i, j] = agg(on_r)["out"] - agg(off_r)["out"]

    fig, ax = plt.subplots(figsize=(6, 6.5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=6.5)
    plt.colorbar(im, ax=ax, label="Graph Gain (dB)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center",
                    fontsize=9, color="black" if 1<matrix[i,j]<4.5 else "white",
                    fontweight="bold")

    ax.set_xticks(range(len(split_names)));    ax.set_xticklabels(split_labels, fontsize=9)
    ax.set_yticks(range(len(model_labels)));   ax.set_yticklabels(model_labels, fontsize=9)
    ax.set_title("Graph Gain Heatmap\n(ON − OFF SI-SDR, dB)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig4_gain_heatmap.pdf")
    fig.savefig(p); fig.savefig(p.replace(".pdf",".png")); plt.close(fig)
    print(f"Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Delta vs input SI-SDR scatter (shows robustness)
# ─────────────────────────────────────────────────────────────────────────────

def fig_delta_vs_input(data, out_dir):
    """Binned-mean ± 1 std curves: ΔSI-SDR vs input SI-SDR (no scatter clutter)."""
    split_name = "Worst"
    models_to_plot = [
        ("Local (IRM)",      COLORS["OFF"],      "Local (IRM)"),
        ("Graph-All (IRM)",  COLORS["ON_all"],   "Graph-All (IRM)"),
        ("Graph-F (PSM)",    COLORS["V9_ON"],    "Graph-F (PSM)"),
        ("Graph-All (PSM)",  COLORS["V9_all"],   "Graph-All (PSM)"),
        ("Graph-All (cIRM)", COLORS["V10_all"],  "Graph-All (cIRM)"),
    ]
    n_bins = 10
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m_name, color, label in models_to_plot:
        rows = data[split_name].get(m_name, [])
        if not rows: continue
        xi = np.array([r["sdr_in"] for r in rows])
        yi = np.array([r["delta"]  for r in rows])
        # equal-frequency bins
        order  = np.argsort(xi)
        xi_s, yi_s = xi[order], yi[order]
        bins   = np.array_split(np.arange(len(xi_s)), n_bins)
        bx     = np.array([xi_s[b].mean() for b in bins])
        bm     = np.array([yi_s[b].mean() for b in bins])
        bs     = np.array([yi_s[b].std()  for b in bins])
        ax.plot(bx, bm, color=color, linewidth=2, label=label, marker="o", markersize=4)
        ax.fill_between(bx, bm - bs, bm + bs, color=color, alpha=0.15)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Input SI-SDR (dB)")
    ax.set_ylabel("ΔSI-SDR (dB)")
    ax.set_title("Enhancement vs. Input Quality\n[Worst split: SNR −8–0 dB]\n(binned mean ± 1 std)")
    ax.legend(fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig5_delta_vs_input.pdf")
    fig.savefig(p); fig.savefig(p.replace(".pdf",".png")); plt.close(fig)
    print(f"Saved {p}")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX tables
# ─────────────────────────────────────────────────────────────────────────────

def latex_main_table(data, out_dir):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Overall SI-SDR (dB) across test conditions. "
                 r"$\Delta$ = SI-SDR$_\text{out}$ $-$ SI-SDR$_\text{in}$. "
                 r"Graph gain = ON $-$ OFF (matched mask type).}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lcc" + "ccc"*3 + r"}")
    lines.append(r"\toprule")
    lines.append(r" & & & \multicolumn{3}{c}{\textbf{Standard}} "
                 r"& \multicolumn{3}{c}{\textbf{Hard}} "
                 r"& \multicolumn{3}{c}{\textbf{Worst}} \\")
    lines.append(r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}")
    lines.append(r"Model & Graph & Mask "
                 r"& in & out & $\Delta$ "
                 r"& in & out & $\Delta$ "
                 r"& in & out & $\Delta$ \\")
    lines.append(r"\midrule")

    def cell(rows):
        if not rows: return "-- & -- & --"
        v = agg(rows)
        return f"{v['in']:.2f} & {v['out']:.2f} & ${v['delta']:+.2f}$"

    # group by OFF/ON
    prev_graph = None
    for short_name, graph, mask, adj, _, run_dir, _ in MODEL_REGISTRY:
        if prev_graph is not None and prev_graph != graph:
            lines.append(r"\midrule")
        prev_graph = graph

        g_str = r"\checkmark" if graph == "ON" else "—"
        adj_str = f" ({adj})" if adj not in ("—","fixed") else ""
        name_str = f"{short_name}{adj_str}".replace("_", r"\_")

        cells = " & ".join(cell(data[s[0]].get(short_name, [])) for s in SPLITS)
        lines.append(f"{name_str} & {g_str} & {mask} & {cells} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    p = os.path.join(out_dir, "table_main.tex")
    with open(p, "w") as f: f.write("\n".join(lines))
    print(f"Saved {p}")


def latex_graph_gain_table(data, out_dir):
    pairs = [
        ("Graph-F (IRM)",    "Local (IRM)",  "Graph-F (IRM, fixed adj.)"),
        ("Graph-F+ (IRM)",   "Local (IRM)",  "Graph-F+ (IRM, fixed, 120ep)"),
        ("Graph-All (IRM)",  "Local (IRM)",  "Graph-All (IRM, all-to-all)"),
        ("Graph-L1 (IRM)",   "Local (IRM)",  "Graph-L1 (IRM, L1-inject)"),
        ("Graph-F (PSM)",    "Local (PSM)",  "Graph-F (PSM, fixed adj.)"),
        ("Graph-All (PSM)",  "Local (PSM)",  "Graph-All (PSM, all-to-all)"),
        ("Graph-F (cIRM)",   "Local (cIRM)", "Graph-F (cIRM, fixed adj.)"),
        ("Graph-All (cIRM)", "Local (cIRM)", "Graph-All (cIRM, all-to-all)"),
    ]
    split_names  = [s[0] for s in SPLITS]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Graph gain (ON $-$ OFF SI-SDR, dB) by condition. "
                 r"Best result per column in \textbf{bold}.}")
    lines.append(r"\label{tab:graph_gain}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Standard & Hard & Worst \\")
    lines.append(r"\midrule")

    # compute all gains first to find max per column
    all_gains = {}
    for on_n, off_n, label in pairs:
        gains = []
        for split_name in split_names:
            on_r  = data[split_name].get(on_n,  [])
            off_r = data[split_name].get(off_n, [])
            g = agg(on_r)["out"] - agg(off_r)["out"] if (on_r and off_r) else float("nan")
            gains.append(g)
        all_gains[on_n] = gains

    col_max = [max(all_gains[on_n][j] for on_n,_,_ in pairs
                   if not np.isnan(all_gains[on_n][j]))
               for j in range(len(split_names))]

    for on_n, off_n, label in pairs:
        gains = all_gains[on_n]
        cells = []
        for j, g in enumerate(gains):
            s = f"${g:+.2f}$"
            if abs(g - col_max[j]) < 0.001: s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(f"{label.replace('_', r'_')} & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    p = os.path.join(out_dir, "table_graph_gain.tex")
    with open(p, "w") as f: f.write("\n".join(lines))
    print(f"Saved {p}")


def latex_breakdown_table(data, out_dir):
    """Breakdown by spk/snr/rt60 for worst-case split, key models only."""
    split_name  = "Worst"
    key_models  = [
        "Local (IRM)", "Graph-F (IRM)", "Graph-All (IRM)",
        "Graph-F (PSM)", "Graph-All (PSM)",
        "Graph-F (cIRM)", "Graph-All (cIRM)",
    ]
    key_labels  = [
        "Local (IRM)", "G-F (IRM)", "G-All (IRM)",
        "G-F (PSM)", "G-All (PSM)",
        "G-F (cIRM)", "G-All (cIRM)",
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{$\Delta$SI-SDR breakdown on Worst-case split "
                 r"(SNR $-8$--$0$\,dB, RT60 0.6--0.9\,s, 3 spk/table). "
                 r"Best graph model per row in \textbf{bold}.}")
    lines.append(r"\label{tab:breakdown_worst}")
    lines.append(r"\begin{tabular}{l" + "c"*len(key_models) + "}")
    lines.append(r"\toprule")
    head = "Condition & " + " & ".join(key_labels) + r" \\"
    lines.append(head)
    lines.append(r"\midrule")

    for key, label, groups in [
        ("g_spk",  "Speaker count", None),
        ("g_snr",  "SNR tier",      None),
        ("g_rt60", "RT60 tier",     None),
    ]:
        # collect all groups for this key
        all_groups = sorted({r[key] for m in key_models
                             for r in data[split_name].get(m, [])})
        lines.append(r"\multicolumn{" + str(len(key_models)+1) + r"}{l}{"
                     r"\textit{" + label + r"}} \\")
        for g in all_groups:
            vals = []
            for m_name in key_models:
                rows = data[split_name].get(m_name, [])
                bd   = breakdown(rows, key)
                v    = bd.get(g, {}).get("delta", float("nan"))
                vals.append(v)
            # bold best ON model (skip OFF at index 0)
            on_vals = [v for i,v in enumerate(vals) if i > 0 and not np.isnan(v)]
            best_on = max(on_vals) if on_vals else float("nan")
            cells = []
            for i, v in enumerate(vals):
                s = f"${v:+.2f}$" if not np.isnan(v) else "--"
                if i > 0 and abs(v - best_on) < 0.001: s = r"\textbf{" + s + "}"
                cells.append(s)
            g_short = g.replace("≤","-").replace("≥","$\\geq$").replace("<","$<$").replace(">","$>$")
            lines.append(f"\\quad {g_short} & {' & '.join(cells)} \\\\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"  # replace last midrule
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    p = os.path.join(out_dir, "table_breakdown_worst.tex")
    with open(p, "w") as f: f.write("\n".join(lines))
    print(f"Saved {p}")


def latex_absolute_table(data, out_dir):
    """Absolute SI-SDR table: input / output / delta for all models × 3 splits."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Absolute SI-SDR (dB) results across all models and test conditions. "
                 r"$\Delta$ = SI-SDR$_\text{out}$ $-$ SI-SDR$_\text{in}$. "
                 r"Best $\Delta$ per condition column is \textbf{bold}.}")
    lines.append(r"\label{tab:absolute}")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{llcc" + "ccc"*3 + r"}")
    lines.append(r"\toprule")
    lines.append(r" & & & & \multicolumn{3}{c}{\textbf{Standard}} "
                 r"& \multicolumn{3}{c}{\textbf{Hard}} "
                 r"& \multicolumn{3}{c}{\textbf{Worst}} \\")
    lines.append(r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}")
    lines.append(r"Model & Mask & Graph & Adj "
                 r"& in & out & $\Delta$ "
                 r"& in & out & $\Delta$ "
                 r"& in & out & $\Delta$ \\")
    lines.append(r"\midrule")

    def cell(rows):
        if not rows: return "-- & -- & --"
        v = agg(rows)
        return f"{v['in']:.1f} & {v['out']:.1f} & ${v['delta']:+.2f}$"

    # compute best delta per split for bolding
    split_names = [s[0] for s in SPLITS]
    best_delta = {}
    for j, sn in enumerate(split_names):
        vals = []
        for short_name, *_ in MODEL_REGISTRY:
            rows = data[sn].get(short_name, [])
            if rows: vals.append(agg(rows)["delta"])
        best_delta[sn] = max(vals) if vals else float("nan")

    prev_mask = None
    for short_name, graph, mask, adj, _, run_dir, _ in MODEL_REGISTRY:
        if prev_mask is not None and prev_mask != mask:
            lines.append(r"\midrule")
        prev_mask = mask

        g_str   = r"\checkmark" if graph == "ON" else "—"
        adj_str = adj if adj != "—" else "—"
        name_str = short_name.replace("(", r"\textit{(").replace(")", r")}") \
                              .replace("_", r"\_")

        row_cells = []
        for j, sn in enumerate(split_names):
            rows = data[sn].get(short_name, [])
            if not rows:
                row_cells.append("-- & -- & --")
                continue
            v = agg(rows)
            delta_str = f"${v['delta']:+.2f}$"
            if abs(v["delta"] - best_delta[sn]) < 0.01:
                delta_str = r"\textbf{" + delta_str + "}"
            row_cells.append(f"{v['in']:.1f} & {v['out']:.1f} & {delta_str}")

        lines.append(f"{short_name} & {mask} & {g_str} & {adj_str} & "
                     f"{' & '.join(row_cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    p = os.path.join(out_dir, "table_absolute.tex")
    with open(p, "w") as f: f.write("\n".join(lines))
    print(f"Saved {p}")


def save_csv(data, out_dir):
    """Save all per-scene rows as CSV for further analysis."""
    csv_dir = os.path.join(out_dir, "csv")
    _ensure(csv_dir)
    for split_name, _ , _ in SPLITS:
        for short_name, *_ in MODEL_REGISTRY:
            rows = data[split_name].get(short_name, [])
            if not rows: continue
            p = os.path.join(csv_dir, f"{split_name}_{short_name}.csv")
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
    print(f"CSVs saved → {csv_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",     default="/home/rrame12/Desktop/Research/ASN/paper_output")
    ap.add_argument("--batch",       type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    fig_dir   = os.path.join(args.out_dir, "figures")
    table_dir = os.path.join(args.out_dir, "tables")
    _ensure(fig_dir); _ensure(table_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Evaluate / load cached per-scene data ──────────────────────────
    print("\n=== Loading / evaluating models ===")
    data = load_all_data(args.out_dir, device, args.batch, args.num_workers)

    # ── 2. Figures ────────────────────────────────────────────────────────
    print("\n=== Generating figures ===")
    fig_graph_gain_progression(data, fig_dir)
    fig_boxplots(data, fig_dir)
    fig_combined_breakdown(data, fig_dir)
    fig_gain_heatmap(data, fig_dir)
    fig_delta_vs_input(data, fig_dir)

    # ── 3. LaTeX tables ───────────────────────────────────────────────────
    print("\n=== Generating LaTeX tables ===")
    latex_main_table(data, table_dir)
    latex_graph_gain_table(data, table_dir)
    latex_breakdown_table(data, table_dir)
    latex_absolute_table(data, table_dir)

    # ── 4. CSVs ───────────────────────────────────────────────────────────
    save_csv(data, args.out_dir)

    # ── 5. Print summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  All outputs saved to: {args.out_dir}/")
    print(f"  figures/   — 5 PDF + PNG figures")
    print(f"  tables/    — 4 LaTeX .tex files")
    print(f"  per_scene/ — cached per-scene JSON")
    print(f"  csv/       — per-scene CSV files")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
