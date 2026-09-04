#!/usr/bin/env python3
"""Plot SI-SDR versus communication bandwidth for the IWAENC paper."""

import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_asn")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache_asn")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPLITS = ["Standard", "Hard", "Worst"]
SPLIT_LABELS = {
    "Standard": "SNR 5-20 dB",
    "Hard": "SNR 0-8 dB",
    "Worst": "SNR <0 dB",
}
COLORS = {
    "Standard": "#2F6690",
    "Hard": "#C97C1A",
    "Worst": "#B33A3A",
}

# Equivalent per-node payloads from the current draft, in Mb/s.
# Local uses no cross-node communication. Bottleneck sharing uses the
# low-rate bottleneck feature path. Neighbourhood and fully shared networks
# include input-level sharing plus bottleneck feature sharing.
MODELS = [
    ("local\n0 Mb/s", "Local (IRM)", 0.0),
    ("Bottleneck\nfeature sharing\n0.069 Mb/s", "Bottleneck+FiLM (V7)", 0.0348 + 0.0338),
    ("Neighbourhood\nshared network\n1.06 Mb/s", "Graph-F (IRM)", 1.03 + 0.0348),
    ("fully shared\nnetwork\n3.19 Mb/s", "Graph-All (IRM)", 3 * (1.03 + 0.0348)),
]


def safe_name(name):
    return name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")


def load_rows(cache_dir, split, model):
    candidates = [
        f"{split}_{model}.json",
        f"{split}_{safe_name(model)}.json",
    ]
    for fname in candidates:
        path = os.path.join(cache_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)["rows"]
    return []


def mean(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else None


def mean_std(rows, key):
    vals = np.array([r[key] for r in rows if key in r and r[key] == r[key]], dtype=float)
    if vals.size == 0:
        return None, None
    return float(vals.mean()), float(vals.std(ddof=0))


def plot_tradeoff(cache_dir, out_dir):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(8.1, 4.9))

    for split in SPLITS:
        xs, ys, yerr = [], [], []
        for _, model, mbps in MODELS:
            rows = load_rows(cache_dir, split, model)
            val, std = mean_std(rows, "sdr_out")
            if val is None:
                continue
            xs.append(mbps)
            ys.append(val)
            yerr.append(std)

        xs = np.array(xs)
        ys = np.array(ys)
        yerr = np.array(yerr)

        ax.fill_between(
            xs,
            ys - yerr,
            ys + yerr,
            color=COLORS[split],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=6,
            linewidth=2.0,
            color=COLORS[split],
            label=SPLIT_LABELS[split],
            zorder=3,
        )

    ax.set_xscale("symlog", linthresh=0.04)
    ax.set_xticks([m[2] for m in MODELS])
    ax.set_xticklabels([m[0] for m in MODELS], fontsize=7.5, rotation=28, ha="right")
    ax.set_xlabel("Equivalent cross-node payload per node (Mb/s)")
    ax.set_ylabel("Output SI-SDR (dB)")
    ax.set_title("Performance vs Communication Bandwidth")
    ax.legend(framealpha=0.9, loc="lower right")
    ax.margins(x=0.10, y=0.20)
    fig.subplots_adjust(bottom=0.30)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "fig11_bandwidth_tradeoff")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png")
    plt.close(fig)
    print(f"Saved {base}.pdf/.png")


def paired_gain(rows, local_rows):
    n = min(len(rows), len(local_rows))
    vals = []
    for i in range(n):
        if "sdr_out" not in rows[i] or "sdr_out" not in local_rows[i]:
            continue
        vals.append(rows[i]["sdr_out"] - local_rows[i]["sdr_out"])
    vals = np.array(vals, dtype=float)
    if vals.size == 0:
        return None, None
    return float(vals.mean()), float(vals.std(ddof=0))


def plot_gain_tradeoff(cache_dir, out_dir):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(8.1, 4.9))

    for split in SPLITS:
        local_rows = load_rows(cache_dir, split, "Local (IRM)")
        xs, ys, yerr = [], [], []
        for _, model, mbps in MODELS:
            rows = load_rows(cache_dir, split, model)
            if model == "Local (IRM)":
                val, std = 0.0, 0.0
            else:
                val, std = paired_gain(rows, local_rows)
            if val is None:
                continue
            xs.append(mbps)
            ys.append(val)
            yerr.append(std)

        xs = np.array(xs)
        ys = np.array(ys)
        yerr = np.array(yerr)

        ax.fill_between(
            xs,
            ys - yerr,
            ys + yerr,
            color=COLORS[split],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=6,
            linewidth=2.0,
            color=COLORS[split],
            label=SPLIT_LABELS[split],
            zorder=3,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.8)
    ax.set_xscale("symlog", linthresh=0.04)
    ax.set_xticks([m[2] for m in MODELS])
    ax.set_xticklabels([m[0] for m in MODELS], fontsize=7.5, rotation=28, ha="right")
    ax.set_xlabel("Equivalent cross-node payload per node (Mb/s)")
    ax.set_ylabel("SI-SDR gain over local (dB)")
    ax.set_title("Communication Benefit vs Bandwidth")
    ax.legend(framealpha=0.9, loc="upper left")
    ax.margins(x=0.10, y=0.18)
    fig.subplots_adjust(bottom=0.30)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "fig12_bandwidth_gain_tradeoff")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png")
    plt.close(fig)
    print(f"Saved {base}.pdf/.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default="/home/rrame12/Desktop/Research/ASN/paper_output/per_scene",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(__file__), "generated_figures"),
    )
    args = parser.parse_args()
    plot_tradeoff(args.cache_dir, args.out_dir)
    plot_gain_tradeoff(args.cache_dir, args.out_dir)


if __name__ == "__main__":
    main()
