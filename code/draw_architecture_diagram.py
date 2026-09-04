#!/usr/bin/env python3
"""
draw_architecture_diagram.py

Vector architecture diagram for the ASN paper.

Outputs:
  fig0_architecture.pdf
  fig0_architecture.png

Usage:
  python draw_architecture_diagram.py \
      --out_dir /home/rrame12/Desktop/Research/ASN/paper_output/figures_v2
"""

import os
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_asn")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache_asn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


COL = {
    "ink":      "#243447",
    "muted":    "#6B7C93",
    "local":    "#DCEBFA",
    "shared":   "#E1F3E6",
    "context":  "#FFF1CC",
    "bott":     "#F7D9D9",
    "output":   "#E9ECEF",
    "group":    "#F8FAFC",
    "accent":   "#2F6690",
    "accent2":  "#2D936C",
    "accent3":  "#C97C1A",
}


def add_box(ax, x, y, w, h, text, fc, ec=None, lw=1.4, fontsize=10.0, weight="normal"):
    if ec is None:
        ec = COL["ink"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COL["ink"],
        linespacing=1.12,
        fontweight=weight,
    )
    return patch


def add_label(ax, x, y, text, fc, fontsize=10.0):
    add_box(ax, x, y, 2.5, 0.52, text, fc=fc, ec=fc, lw=0.0, fontsize=fontsize, weight="bold")


def add_arrow(ax, start, end, text=None, color=None, lw=1.7, linestyle="-", rad=0.0, text_offset=(0.0, 0.0), fontsize=9.5):
    if color is None:
        color = COL["ink"]
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(arrow)
    if text:
        mx = 0.5 * (start[0] + end[0]) + text_offset[0]
        my = 0.5 * (start[1] + end[1]) + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fontsize, color=color)
    return arrow


def draw_diagram(out_dir):
    fig, ax = plt.subplots(figsize=(16.5, 6.8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Group labels
    add_label(ax, 0.8, 9.0, "Node-Local Path", fc=COL["local"])
    add_label(ax, 4.0, 9.0, "Cross-Node Sharing", fc=COL["shared"])
    add_label(ax, 12.8, 9.0, "Bottleneck Conditioning", fc=COL["context"])

    # Main flow boxes
    wave = add_box(ax, 0.6, 4.25, 2.0, 1.45, "Node $k$ mixture\n$x_k(t)$", fc=COL["output"], fontsize=10.5)
    feats = add_box(
        ax,
        3.1,
        4.05,
        2.95,
        1.85,
        "STFT + local features\n$[\\log(1+|X_k|),\\; \\mathrm{Re}/|X_k|,\\; \\mathrm{Im}/|X_k|]$",
        fc=COL["local"],
        fontsize=9.6,
    )
    concat = add_box(ax, 6.65, 4.25, 2.05, 1.45, "Concatenate\nlocal + $Z_k$", fc=COL["shared"], fontsize=10.2)

    # Shared U-Net group
    backbone = FancyBboxPatch(
        (9.25, 3.2),
        7.15,
        3.55,
        boxstyle="round,pad=0.03,rounding_size=0.16",
        facecolor=COL["group"],
        edgecolor="#D7DEE8",
        linewidth=1.1,
    )
    ax.add_patch(backbone)
    ax.text(12.83, 6.22, "Shared U-Net Backbone", ha="center", va="center",
            fontsize=11.2, fontweight="bold", color=COL["ink"])

    enc = add_box(ax, 9.75, 3.8, 1.45, 2.25, "Encoder", fc=COL["local"], fontsize=10.8, weight="bold")
    bott = add_box(ax, 11.75, 4.45, 1.9, 0.95, "Bottleneck\n$h_k$", fc=COL["bott"], fontsize=10.3, weight="bold")
    dec = add_box(ax, 14.15, 3.8, 1.45, 2.25, "Decoder", fc=COL["local"], fontsize=10.8, weight="bold")
    mask = add_box(ax, 16.0, 4.2, 1.7, 1.55, "Mask head\nIRM / PSM / cIRM", fc=COL["context"], fontsize=9.6)
    recon = add_box(ax, 18.1, 4.2, 1.55, 1.55, "Apply mask\n+ iSTFT", fc=COL["output"], fontsize=9.8)

    ax.text(19.7, 4.97, "Enhanced waveform\n$\\hat{s}_k(t)$", ha="left", va="center",
            fontsize=10.8, color=COL["ink"])

    # Top-left shared input branch
    nb = add_box(
        ax,
        3.1,
        7.28,
        2.95,
        1.3,
        "Adjacent nodes $j \\in \\mathcal{N}(k)$\nSTFT + $\\log(1+|X_j|)$",
        fc=COL["shared"],
        fontsize=9.6,
    )
    agg = add_box(
        ax,
        6.65,
        7.18,
        2.5,
        1.5,
        "Adjacency-weighted\nneighbor log-mag\n$Z_k = \\sum_j \\tilde{A}_{k,j}\\,\\log(1+|X_j|)$",
        fc=COL["shared"],
        fontsize=9.1,
    )

    # Top-right bottleneck branch
    collect = add_box(
        ax,
        11.35,
        7.28,
        2.2,
        1.3,
        "Collect $\\{h_j\\}_{j=1}^{K}$\nmean over time",
        fc=COL["context"],
        fontsize=9.7,
    )
    attn = add_box(
        ax,
        14.05,
        7.18,
        3.1,
        1.5,
        "Per-frequency\ncross-node attention\n+ FiLM modulation",
        fc=COL["context"],
        fontsize=9.5,
    )

    # Main arrows
    add_arrow(ax, (2.6, 4.97), (3.1, 4.97))
    add_arrow(ax, (6.05, 4.97), (6.65, 4.97))
    add_arrow(ax, (8.7, 4.97), (9.75, 4.97))
    add_arrow(ax, (11.2, 4.97), (11.75, 4.97))
    add_arrow(ax, (13.65, 4.97), (14.15, 4.97))
    add_arrow(ax, (15.6, 4.97), (16.0, 4.97))
    add_arrow(ax, (17.7, 4.97), (18.1, 4.97))
    add_arrow(ax, (19.6, 4.97), (20.0, 4.97))

    # Shared-input branch arrows
    add_arrow(ax, (6.05, 7.93), (6.65, 7.93), text="$A$", color=COL["accent2"], text_offset=(0.0, 0.38))
    add_arrow(ax, (7.9, 7.18), (7.7, 5.7), text="input injection", color=COL["accent2"], text_offset=(0.68, 0.1))

    # Bottleneck branch arrows
    add_arrow(ax, (12.7, 5.4), (12.45, 7.28), color=COL["accent3"])
    add_arrow(ax, (13.55, 7.93), (14.05, 7.93), text="$A$", color=COL["accent3"], text_offset=(0.0, 0.38))
    add_arrow(ax, (15.6, 7.18), (12.7, 5.35), color=COL["accent3"], rad=-0.08)

    # Skip connection hint
    add_arrow(
        ax,
        (11.2, 5.9),
        (14.15, 5.9),
        text=None,
        color=COL["muted"],
        lw=1.3,
        linestyle="--",
        rad=0.35,
    )

    # Notes
    ax.text(
        0.75,
        2.2,
        "Only log-magnitude is shared across nodes at the input because inter-node phase is not aligned.\n"
        "Graph OFF removes the $Z_k$ input branch and the bottleneck FiLM path, yielding a strict local-only U-Net.",
        ha="left",
        va="center",
        fontsize=9.7,
        color=COL["ink"],
    )

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "fig0_architecture")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png")
    plt.close(fig)
    print(f"Saved {base}.pdf/.png")


def main():
    default_out = os.path.join(os.path.dirname(__file__), "generated_figures")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        default=default_out,
        help="Output directory for architecture figure.",
    )
    args = ap.parse_args()
    draw_diagram(args.out_dir)


if __name__ == "__main__":
    main()
