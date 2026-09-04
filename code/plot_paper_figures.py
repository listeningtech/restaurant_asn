#!/usr/bin/env python3
"""
plot_paper_figures.py  —  Publication-quality figures for the ASN paper.

Figures generated:
  fig1_overall_metrics.pdf       — Grouped bars: SI-SDR/PESQ/STOI for all 9 models × 3 splits
  fig2_graph_gain_progression.pdf — Graph gain (ΔSI-SDR, ΔPESQ, ΔSTOI) vs acoustic difficulty
  fig3_breakdown_snr.pdf         — SI-SDR/PESQ/STOI broken down by SNR tier (3 panels)
  fig4_breakdown_rt60.pdf        — Breakdown by RT60 tier
  fig5_breakdown_spk.pdf         — Breakdown by speaker count
  fig6_radar_worst.pdf           — Radar chart: all 9 models, 3 metrics on Worst split
  fig7_delta_vs_input_binned.pdf — Binned ΔSI-SDR vs input SI-SDR curves (Worst split)
  fig8_arch_ablation.pdf         — Architecture ablation: V7 vs V8 bar chart
  fig9_heatmap_sisdr.pdf         — SI-SDR heatmap: models × conditions

Usage:
  python plot_paper_figures.py \\
      --cache_dir /home/rrame12/Desktop/Research/ASN/paper_output/per_scene \\
      --out_dir   /home/rrame12/Desktop/Research/ASN/paper_output/figures_v2
"""

import os, sys, json, argparse
from collections import defaultdict
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# ── Paper style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.axisbelow":    True,
})

# ── Palette ───────────────────────────────────────────────────────────────────
# 3 mask types × 3 variants (Local / G-F / G-All)
PAL = {
    # IRM — blue family
    "Local (IRM)":      "#9ECAE1",
    "Graph-F (IRM)":    "#3182BD",
    "Graph-All (IRM)":  "#08306B",
    # PSM — orange family
    "Local (PSM)":      "#FDAE6B",
    "Graph-F (PSM)":    "#E6550D",
    "Graph-All (PSM)":  "#7F2704",
    # cIRM — green family
    "Local (cIRM)":     "#A1D99B",
    "Graph-F (cIRM)":   "#31A354",
    "Graph-All (cIRM)": "#00441B",
}

# Short display names for tick labels
SHORT = {
    "Local (IRM)":      "Loc\n(IRM)",
    "Graph-F (IRM)":    "G-F\n(IRM)",
    "Graph-All (IRM)":  "G-All\n(IRM)",
    "Local (PSM)":      "Loc\n(PSM)",
    "Graph-F (PSM)":    "G-F\n(PSM)",
    "Graph-All (PSM)":  "G-All\n(PSM)",
    "Local (cIRM)":     "Loc\n(cIRM)",
    "Graph-F (cIRM)":   "G-F\n(cIRM)",
    "Graph-All (cIRM)": "G-All\n(cIRM)",
}

MODELS = list(PAL.keys())
SPLITS  = ["Standard", "Hard", "Worst"]
SPLIT_LABELS = ["Standard\n(SNR 5–20 dB)", "Hard\n(SNR 0–8 dB)", "Worst\n(SNR <0 dB)"]

# Pairs for graph-gain computation
GAIN_PAIRS = [
    ("Graph-F (IRM)",    "Local (IRM)"),
    ("Graph-All (IRM)",  "Local (IRM)"),
    ("Graph-F (PSM)",    "Local (PSM)"),
    ("Graph-All (PSM)",  "Local (PSM)"),
    ("Graph-F (cIRM)",   "Local (cIRM)"),
    ("Graph-All (cIRM)", "Local (cIRM)"),
]
GAIN_COLORS = ["#3182BD","#08306B","#E6550D","#7F2704","#31A354","#00441B"]
GAIN_LABELS = ["G-F (IRM)","G-All (IRM)","G-F (PSM)","G-All (PSM)","G-F (cIRM)","G-All (cIRM)"]

# ── Data helpers ──────────────────────────────────────────────────────────────

def _nm(vals):
    v = [x for x in vals if x == x]
    return sum(v)/len(v) if v else float("nan")

def agg(rows):
    if not rows: return None
    n = len(rows)
    return {
        "sdr_in":   sum(r["sdr_in"]  for r in rows)/n,
        "sdr_out":  sum(r["sdr_out"] for r in rows)/n,
        "delta":    sum(r["delta"]   for r in rows)/n,
        "pesq_in":  _nm([r.get("pesq_in",  float("nan")) for r in rows]),
        "pesq_out": _nm([r.get("pesq_out", float("nan")) for r in rows]),
        "stoi_in":  _nm([r.get("stoi_in",  float("nan")) for r in rows]),
        "stoi_out": _nm([r.get("stoi_out", float("nan")) for r in rows]),
        "n": n,
    }

def load_cache(cache_dir):
    """Returns data[split][model] = list[dict]."""
    data = {s: {} for s in SPLITS}
    # Also include V7 ablation models
    all_models = MODELS + ["Bottleneck-only (V7)", "Bottleneck+FiLM (V7)"]
    legacy = {
        "Local (IRM)":     "V8-OFF",  "Graph-F (IRM)":    "V8-ON",
        "Graph-All (IRM)": "V8-alltoall",
        "Local (PSM)":     "V9-OFF",  "Graph-F (PSM)":    "V9-ON",
        "Local (cIRM)":    "V10-OFF", "Graph-F (cIRM)":   "V10-ON",
    }
    def safe(s): return s.replace("/","_").replace(" ","_").replace("(","").replace(")","")

    for split in SPLITS:
        for m in all_models:
            # try new safe name, then name with spaces/parens, then legacy
            for fname in [
                f"{split}_{safe(m)}.json",
                f"{split}_{m}.json",
                f"{split}_{legacy.get(m,'__none__')}.json",
            ]:
                p = os.path.join(cache_dir, fname)
                if os.path.exists(p):
                    data[split][m] = json.load(open(p))["rows"]
                    break
    return data


def pool(data, models, splits, snr_fn=None, rt60_fn=None, spk=None):
    """Pool rows across splits, optionally filtered."""
    out = defaultdict(list)
    for s in splits:
        for m in models:
            for r in data[s].get(m, []):
                if snr_fn  and not snr_fn(r["snr"]):   continue
                if rt60_fn and not rt60_fn(r["rt60"]): continue
                if spk is not None and r["spk"] != spk: continue
                out[m].append(r)
    return out

def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(out_dir, name + ext))
    plt.close(fig)
    print(f"  Saved {name}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Overall metrics: 3×3 grouped-bar grid (split × metric)
# ─────────────────────────────────────────────────────────────────────────────

def fig_overall_metrics(data, out_dir):
    metrics  = [("sdr_out","SI-SDR (dB)"), ("pesq_out","PESQ (WB-MOS)"), ("stoi_out","STOI")]
    mix_keys = ["sdr_in", "pesq_in", "stoi_in"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharey="col")
    fig.subplots_adjust(hspace=0.45, wspace=0.28)

    x  = np.arange(len(MODELS))
    bw = 0.65

    for row, (split, slabel) in enumerate(zip(SPLITS, SPLIT_LABELS)):
        for col, ((mkey, mlabel), mixin) in enumerate(zip(metrics, mix_keys)):
            ax = axes[row][col]
            vals  = [agg(data[split].get(m,[]))[mkey] if agg(data[split].get(m,[])) else 0
                     for m in MODELS]
            mix_v = agg(data[split].get(MODELS[0],[]))[mixin] if agg(data[split].get(MODELS[0],[])) else None
            colors = [PAL[m] for m in MODELS]
            bars = ax.bar(x, vals, bw, color=colors, edgecolor="white", linewidth=0.6, zorder=3)

            # Annotate best
            best = max(vals)
            for bar, v in zip(bars, vals):
                if abs(v - best) < 0.005:
                    ax.annotate("*", xy=(bar.get_x()+bar.get_width()/2, v),
                                xytext=(0, 2), textcoords="offset points",
                                ha="center", fontsize=12, color="#CC0000",
                                fontweight="bold")

            # Mixture baseline line
            if mix_v is not None:
                ax.axhline(mix_v, color="#555", linewidth=1.2,
                           linestyle=":", label="Mixture")

            # Vertical separators between mask groups
            for sep in [2.5, 5.5]:
                ax.axvline(sep, color="#bbb", linewidth=0.8, linestyle="--", zorder=2)

            ax.set_xticks(x)
            ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=7.5)
            ax.set_ylabel(mlabel, fontsize=10)
            if row == 0:
                ax.set_title(mlabel, fontsize=11, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(f"{slabel}\n\n{mlabel}", fontsize=9)
            else:
                ax.set_ylabel("")

            # Shade mask-group background
            for span, alpha, fc in [((-0.5, 2.5), 0.06, "#2196F3"),
                                     ((2.5, 5.5),  0.06, "#FF9800"),
                                     ((5.5, 8.5),  0.06, "#4CAF50")]:
                ax.axvspan(span[0], span[1], alpha=alpha, color=fc, zorder=0)

    # Column headers (mask types)
    for col, lbl in enumerate(["IRM", "PSM", "cIRM"]):
        axes[0][col].set_title(f"{lbl}\n{metrics[col][1]}", fontsize=11, fontweight="bold")

    # Row labels (splits) — use text annotation on left edge
    for row, slabel in enumerate(SPLIT_LABELS):
        axes[row][0].set_ylabel(f"{slabel}\n\n{metrics[0][1]}", fontsize=9, labelpad=6)

    # Shared legend
    patches = [mpatches.Patch(color=PAL[m], label=SHORT[m].replace("\n"," ")) for m in MODELS]
    fig.legend(handles=patches, loc="lower center", ncol=9,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Output Quality Metrics Across Models and Test Conditions\n"
                 "(★ = best per column, dotted line = mixture baseline)",
                 fontsize=13, y=1.01, fontweight="bold")
    save(fig, out_dir, "fig1_overall_metrics")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Graph gain progression: ΔSI-SDR / ΔPESQ / ΔSTOI vs difficulty
# ─────────────────────────────────────────────────────────────────────────────

def fig_graph_gain_progression(data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.subplots_adjust(wspace=0.32)

    metrics = [
        ("sdr_out",  "sdr_out",  "ΔSI-SDR (dB)",      False),
        ("pesq_out", "pesq_out", "ΔPESQ (WB-MOS)",     False),
        ("stoi_out", "stoi_out", "ΔSTOI",               False),
    ]

    x = np.arange(len(SPLITS))
    n = len(GAIN_PAIRS)
    width = 0.12
    offsets = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)

    for col, (on_key, off_key, ylabel, _) in enumerate(metrics):
        ax = axes[col]
        for i, ((on_m, off_m), color, label) in enumerate(
                zip(GAIN_PAIRS, GAIN_COLORS, GAIN_LABELS)):
            gains = []
            for split in SPLITS:
                a_on  = agg(data[split].get(on_m,  []))
                a_off = agg(data[split].get(off_m, []))
                if a_on and a_off:
                    gains.append(a_on[on_key] - a_off[off_key])
                else:
                    gains.append(0.0)

            style = "-" if "All" in on_m else "--"
            lw    = 2.2 if "All" in on_m else 1.6
            ax.plot(x, gains, marker="o", markersize=7, color=color,
                    linestyle=style, linewidth=lw, label=label, zorder=3)

            # Value labels on the Worst point
            ax.annotate(f"{gains[-1]:+.2f}",
                        xy=(x[-1], gains[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=7.5, color=color, va="center")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(SPLIT_LABELS, fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")

        # Shade background to show increasing difficulty
        for xi, alpha in [(0, 0.04), (1, 0.08), (2, 0.14)]:
            ax.axvspan(xi-0.45, xi+0.45, alpha=alpha, color="#E53935", zorder=0)

    # One legend for all panels
    handles = [Line2D([0],[0], color=c, lw=2, linestyle="-" if "All" in l else "--",
                      marker="o", markersize=5, label=l)
               for c, l in zip(GAIN_COLORS, GAIN_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Graph Gain vs Acoustic Difficulty  (Graph − Local)",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, out_dir, "fig2_graph_gain_progression")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Breakdown by SNR tier: 3 metrics × 3 SNR tiers
# ─────────────────────────────────────────────────────────────────────────────

def _breakdown_fig(data, filters, tier_labels, fig_name, xlabel, out_dir,
                   title, splits=None):
    """Generic 3-row (metric) × N-col (tier) breakdown for key models."""
    if splits is None:
        splits = SPLITS

    # Focus on the most interesting models
    key_models = [
        "Local (IRM)", "Graph-All (IRM)",
        "Graph-All (PSM)", "Graph-All (cIRM)",
    ]
    key_colors = [PAL[m] for m in key_models]
    key_short  = ["Local\n(IRM)", "G-All\n(IRM)", "G-All\n(PSM)", "G-All\n(cIRM)"]

    metrics = [
        ("sdr_out",  "SI-SDR (dB)"),
        ("pesq_out", "PESQ"),
        ("stoi_out", "STOI"),
    ]

    n_tiers = len(tier_labels)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    fig.subplots_adjust(wspace=0.30)

    x      = np.arange(n_tiers)
    n_m    = len(key_models)
    width  = 0.18
    offsets = np.linspace(-(n_m-1)*width/2, (n_m-1)*width/2, n_m)

    for col, (mkey, mlabel) in enumerate(metrics):
        ax = axes[col]
        for i, (m, color, short) in enumerate(zip(key_models, key_colors, key_short)):
            vals = []
            for fn_kwargs in filters:
                pooled = pool(data, [m], splits, **fn_kwargs)
                a = agg(pooled.get(m, []))
                vals.append(a[mkey] if a and a[mkey]==a[mkey] else 0.0)
            bars = ax.bar(x + offsets[i], vals, width, color=color,
                          edgecolor="white", linewidth=0.5, label=short.replace("\n"," "),
                          zorder=3, alpha=0.92)
            # Annotate top of each bar
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{v:.2f}" if mkey != "stoi_out" else f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90, color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels(tier_labels, fontsize=9.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    handles = [mpatches.Patch(color=c, label=s.replace("\n"," "))
               for c, s in zip(key_colors, key_short)]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    save(fig, out_dir, fig_name)


def fig_breakdown_snr(data, out_dir):
    filters = [
        {"snr_fn": lambda v: v >= 10},
        {"snr_fn": lambda v: 0 <= v < 10},
        {"snr_fn": lambda v: v < 0},
    ]
    _breakdown_fig(data, filters,
                   ["Easy\n(SNR ≥10 dB)", "Medium\n(SNR 0–10 dB)", "Hard\n(SNR <0 dB)"],
                   "fig3_breakdown_snr", "SNR Tier", out_dir,
                   "Performance Breakdown by SNR Tier  (pooled across all splits)")

def fig_breakdown_rt60(data, out_dir):
    filters = [
        {"rt60_fn": lambda v: v < 0.5},
        {"rt60_fn": lambda v: 0.5 <= v < 0.7},
        {"rt60_fn": lambda v: v >= 0.7},
    ]
    _breakdown_fig(data, filters,
                   ["Dry\n(RT60 <0.5 s)", "Mild\n(RT60 0.5–0.7 s)", "Wet\n(RT60 ≥0.7 s)"],
                   "fig4_breakdown_rt60", "RT60 Tier", out_dir,
                   "Performance Breakdown by RT60 Tier  (pooled across all splits)")

def fig_breakdown_spk(data, out_dir):
    filters = [{"spk": 1}, {"spk": 2}, {"spk": 3}]
    _breakdown_fig(data, filters,
                   ["1 Speaker", "2 Speakers", "3 Speakers"],
                   "fig5_breakdown_spk", "Speaker Count", out_dir,
                   "Performance Breakdown by Speaker Count  (Standard + Hard splits)",
                   splits=["Standard", "Hard"])


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Radar chart: 9 models on Worst split, 3 metrics
# ─────────────────────────────────────────────────────────────────────────────

def fig_radar(data, out_dir):
    split = "Worst"
    # Metric ranges for normalisation (min, max)
    ranges = {
        "sdr_out":  (-1, 8),   # SI-SDR dB
        "pesq_out": (1.0, 2.0),
        "stoi_out": (0.60, 0.85),
    }
    metric_labels = ["SI-SDR (dB)", "PESQ", "STOI"]
    mkeys = ["sdr_out", "pesq_out", "stoi_out"]

    def normalise(v, mkey):
        lo, hi = ranges[mkey]
        return (v - lo) / (hi - lo) if v == v else 0.0

    n_metrics = len(mkeys)
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5),
                           subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for m in MODELS:
        a = agg(data[split].get(m, []))
        if not a: continue
        vals = [normalise(a[k], k) for k in mkeys]
        vals += vals[:1]
        lw = 2.5 if "All" in m else (1.8 if "G-F" in m else 1.2)
        ls = "-" if "All" in m else ("--" if "G-F" in m else ":")
        ax.plot(angles, vals, color=PAL[m], linewidth=lw, linestyle=ls,
                label=SHORT[m].replace("\n"," "))
        ax.fill(angles, vals, color=PAL[m], alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="gray")

    # Annotate actual metric values on perimeter
    metric_vals = {k: [agg(data[split].get(m,[]))[k]
                       if agg(data[split].get(m,[])) else float("nan")
                       for m in MODELS] for k in mkeys}
    for j, (mkey, mlabel) in enumerate(zip(mkeys, metric_labels)):
        lo, hi = ranges[mkey]
        ax.annotate(f"[{lo:.1f}–{hi:.1f}]",
                    xy=(angles[j], 1.12), xycoords="data",
                    ha="center", va="center", fontsize=7.5, color="#555")

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=8.5, framealpha=0.9)
    ax.set_title("Model Comparison — Worst Split\n(normalised to [min, max] range)",
                 fontsize=11, fontweight="bold", pad=20)
    ax.grid(color="gray", alpha=0.3)
    save(fig, out_dir, "fig6_radar_worst")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Binned ΔSI-SDR vs input SI-SDR (Worst split)
# ─────────────────────────────────────────────────────────────────────────────

def fig_delta_vs_input(data, out_dir):
    split = "Worst"
    plot_models = [
        "Local (IRM)", "Graph-F (IRM)", "Graph-All (IRM)",
        "Graph-All (PSM)", "Graph-All (cIRM)",
    ]
    n_bins = 12

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for m in plot_models:
        rows = data[split].get(m, [])
        if not rows: continue
        xi = np.array([r["sdr_in"]  for r in rows])
        yi = np.array([r["delta"]   for r in rows])
        idx    = np.argsort(xi)
        xi_s, yi_s = xi[idx], yi[idx]
        bins   = np.array_split(np.arange(len(xi_s)), n_bins)
        bx = np.array([xi_s[b].mean() for b in bins])
        bm = np.array([yi_s[b].mean() for b in bins])
        bs = np.array([yi_s[b].std()  for b in bins])
        lw = 2.5 if "All" in m else 1.8
        ls = "-"  if "All" in m else "--"
        ax.plot(bx, bm, color=PAL[m], linewidth=lw, linestyle=ls,
                marker="o", markersize=5, label=SHORT[m].replace("\n"," "), zorder=3)
        ax.fill_between(bx, bm-bs, bm+bs, color=PAL[m], alpha=0.12, zorder=2)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel("Input SI-SDR (dB)")
    ax.set_ylabel("ΔSI-SDR (dB)")
    ax.set_title("Enhancement Gain vs Input Quality\n"
                 "Worst Split — binned mean ± 1σ",
                 fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    save(fig, out_dir, "fig7_delta_vs_input")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — Architecture ablation bar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_arch_ablation(data, out_dir):
    abl_models = [
        ("Local (IRM)",         "#CCCCCC"),
        ("Bottleneck-only (V7)","#AECDE8"),
        ("Bottleneck+FiLM (V7)","#6BAED6"),
        ("Graph-F (IRM)",       PAL["Graph-F (IRM)"]),
        ("Graph-All (IRM)",     PAL["Graph-All (IRM)"]),
    ]
    split_sel = ["Standard", "Worst"]
    split_labels = ["Standard", "Worst"]
    metrics = [("sdr_out","SI-SDR (dB)"),("pesq_out","PESQ"),("stoi_out","STOI")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.subplots_adjust(wspace=0.32)

    x      = np.arange(len(abl_models))
    bw     = 0.35
    offsets = [-bw/2, bw/2]
    split_colors = ["#1976D2", "#C62828"]

    for col, (mkey, mlabel) in enumerate(metrics):
        ax = axes[col]
        for oi, (split, sc) in enumerate(zip(split_sel, split_colors)):
            vals = []
            for m, _ in abl_models:
                a = agg(data[split].get(m, []))
                vals.append(a[mkey] if a and a[mkey]==a[mkey] else 0.0)
            bars = ax.bar(x + offsets[oi], vals, bw, color=sc,
                          alpha=0.8 if oi==0 else 0.95,
                          edgecolor="white", linewidth=0.6,
                          label=split_labels[oi], zorder=3)
            # Value labels
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{v:.2f}" if mkey!="stoi_out" else f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90, color=sc)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" (IRM)","").replace(" (V7)","(V7)")
                            for m,_ in abl_models], fontsize=8.5, rotation=15, ha="right")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel, fontweight="bold")

        # Shade graph region
        ax.axvspan(2.5, len(abl_models)-0.5, alpha=0.05, color="#388E3C", zorder=0)
        ax.axvline(2.5, color="#888", linewidth=0.8, linestyle="--", zorder=2)
        ax.text(3.0, ax.get_ylim()[0] + 0.02*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                "Input injection →", fontsize=7.5, color="#388E3C", style="italic")

        if col == 0:
            ax.legend(fontsize=9, framealpha=0.9)

    fig.suptitle("Architecture Ablation: V7 (Bottleneck Graph) vs V8 (Input + Bottleneck)",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, out_dir, "fig8_arch_ablation")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 — SI-SDR heatmap: models × conditions (3 splits + breakdowns)
# ─────────────────────────────────────────────────────────────────────────────

def fig_heatmap(data, out_dir):
    row_labels = ["Standard", "Hard", "Worst",
                  "1 spk", "2 spk", "3 spk",
                  "Easy SNR", "Med SNR", "Hard SNR",
                  "Dry RT60", "Mild RT60", "Wet RT60"]

    row_fns = [
        lambda d, m: d["Standard"].get(m, []),
        lambda d, m: d["Hard"].get(m, []),
        lambda d, m: d["Worst"].get(m, []),
        lambda d, m: pool(d, [m], ["Standard","Hard"], spk=1).get(m, []),
        lambda d, m: pool(d, [m], ["Standard","Hard"], spk=2).get(m, []),
        lambda d, m: pool(d, [m], SPLITS, spk=3).get(m, []),
        lambda d, m: pool(d, [m], SPLITS, snr_fn=lambda v: v>=10).get(m, []),
        lambda d, m: pool(d, [m], SPLITS, snr_fn=lambda v: 0<=v<10).get(m, []),
        lambda d, m: pool(d, [m], SPLITS, snr_fn=lambda v: v<0).get(m, []),
        lambda d, m: pool(d, [m], SPLITS, rt60_fn=lambda v: v<0.5).get(m, []),
        lambda d, m: pool(d, [m], SPLITS, rt60_fn=lambda v: 0.5<=v<0.7).get(m, []),
        lambda d, m: pool(d, [m], SPLITS, rt60_fn=lambda v: v>=0.7).get(m, []),
    ]

    matrix = np.full((len(row_labels), len(MODELS)), np.nan)
    for i, fn in enumerate(row_fns):
        for j, m in enumerate(MODELS):
            a = agg(fn(data, m))
            if a: matrix[i, j] = a["sdr_out"]

    # Compute graph gain (relative to matching local baseline)
    local_idx = {m: i for i, m in enumerate(MODELS) if "Local" in m}
    gain_matrix = np.full_like(matrix, np.nan)
    for j, m in enumerate(MODELS):
        if "Local" in m: continue
        mask_type = m.split("(")[1].rstrip(")")
        local_m = f"Local ({mask_type})"
        if local_m in local_idx:
            gain_matrix[:, j] = matrix[:, j] - matrix[:, local_idx[local_m]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.subplots_adjust(wspace=0.05)

    col_labels = [SHORT[m].replace("\n"," ") for m in MODELS]
    groups     = ["← IRM →", "← PSM →", "← cIRM →"]

    # ── Left: absolute SI-SDR ────────────────────────────────────────────────
    im1 = ax1.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=-1, vmax=15)
    plt.colorbar(im1, ax=ax1, label="Output SI-SDR (dB)", shrink=0.8, pad=0.02)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i,j]):
                ax1.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center",
                         fontsize=7.5,
                         color="white" if matrix[i,j] > 10 else "black",
                         fontweight="bold")
    ax1.set_xticks(range(len(MODELS)))
    ax1.set_xticklabels(col_labels, fontsize=8, rotation=30, ha="right")
    ax1.set_yticks(range(len(row_labels)))
    ax1.set_yticklabels(row_labels, fontsize=9)
    ax1.set_title("Output SI-SDR (dB)", fontsize=11, fontweight="bold", pad=10)
    # Group separators
    for sep in [2.5, 5.5]:
        ax1.axvline(sep, color="white", linewidth=2)
    for sep in [2.5, 5.5, 8.5]:
        ax1.axhline(sep, color="white", linewidth=1.5)

    # ── Right: graph gain ────────────────────────────────────────────────────
    # Only show non-Local columns
    non_local_idx = [j for j,m in enumerate(MODELS) if "Local" not in m]
    gm = gain_matrix[:, non_local_idx]
    gm_labels = [col_labels[j] for j in non_local_idx]

    im2 = ax2.imshow(gm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=6.5)
    plt.colorbar(im2, ax=ax2, label="Graph Gain: Graph − Local (dB)", shrink=0.8, pad=0.02)
    for i in range(gm.shape[0]):
        for j in range(gm.shape[1]):
            if not np.isnan(gm[i,j]):
                ax2.text(j, i, f"{gm[i,j]:+.1f}", ha="center", va="center",
                         fontsize=7.5,
                         color="white" if gm[i,j] > 4 else "black",
                         fontweight="bold")
    ax2.set_xticks(range(len(non_local_idx)))
    ax2.set_xticklabels(gm_labels, fontsize=8, rotation=30, ha="right")
    ax2.set_yticks(range(len(row_labels)))
    ax2.set_yticklabels([""] * len(row_labels))
    ax2.set_title("Graph Gain (Graph − Local, dB)", fontsize=11, fontweight="bold", pad=10)
    for sep in [1.5, 3.5]:
        ax2.axvline(sep, color="white", linewidth=2)
    for sep in [2.5, 5.5, 8.5]:
        ax2.axhline(sep, color="white", linewidth=1.5)

    fig.suptitle("SI-SDR Results: Absolute Performance and Graph Gain",
                 fontsize=13, fontweight="bold", y=1.01)
    save(fig, out_dir, "fig9_heatmap_sisdr")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10 — Multi-metric improvement summary: violin/box for worst split
# ─────────────────────────────────────────────────────────────────────────────

def fig_violin_worst(data, out_dir):
    """Violin plots of per-scene ΔSDR distribution on Worst split."""
    split = "Worst"
    key_models = ["Local (IRM)", "Graph-All (IRM)",
                  "Graph-All (PSM)", "Graph-All (cIRM)"]

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(1, len(key_models)+1)

    vp = ax.violinplot(
        [([r["delta"] for r in data[split].get(m,[])] or [0])
         for m in key_models],
        positions=positions, widths=0.65, showmedians=True,
        showextrema=True)

    for i, (pc, m) in enumerate(zip(vp["bodies"], key_models)):
        pc.set_facecolor(PAL[m])
        pc.set_edgecolor("white")
        pc.set_alpha(0.8)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2.5)
    vp["cmins"].set_color("#555"); vp["cmaxes"].set_color("#555")
    vp["cbars"].set_color("#555"); vp["cbars"].set_linewidth(0.8)

    # Overlay mean markers
    for i, m in enumerate(key_models):
        rows = data[split].get(m, [])
        if rows:
            mn = np.mean([r["delta"] for r in rows])
            ax.scatter(i+1, mn, marker="D", s=55, color="white",
                       edgecolor=PAL[m], linewidth=1.8, zorder=5, label="_nolegend_")
            ax.text(i+1, mn+0.15, f"{mn:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=PAL[m])

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT[m].replace("\n"," ") for m in key_models], fontsize=10)
    ax.set_ylabel("ΔSI-SDR per scene (dB)")
    ax.set_title("Per-Scene ΔSI-SDR Distribution — Worst Split\n"
                 "(◆ = mean, thick line = median)", fontweight="bold")
    save(fig, out_dir, "fig10_violin_worst")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="/home/rrame12/Desktop/Research/ASN/paper_output/per_scene")
    ap.add_argument("--out_dir",   default="/home/rrame12/Desktop/Research/ASN/paper_output/figures_v2")
    args = ap.parse_args()

    print("Loading caches...")
    data = load_cache(args.cache_dir)

    # Verify coverage
    for split in SPLITS:
        found = [m for m in MODELS if data[split].get(m)]
        print(f"  {split}: {len(found)}/{len(MODELS)} models loaded")

    print("\nGenerating figures...")
    fig_overall_metrics(data, args.out_dir)
    fig_graph_gain_progression(data, args.out_dir)
    fig_breakdown_snr(data, args.out_dir)
    fig_breakdown_rt60(data, args.out_dir)
    fig_breakdown_spk(data, args.out_dir)
    fig_radar(data, args.out_dir)
    fig_delta_vs_input(data, args.out_dir)
    fig_arch_ablation(data, args.out_dir)
    fig_heatmap(data, args.out_dir)
    fig_violin_worst(data, args.out_dir)

    print(f"\nAll figures saved to: {args.out_dir}/")

if __name__ == "__main__":
    main()
