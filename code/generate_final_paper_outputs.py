#!/usr/bin/env python3
"""Generate final one-mask figures and tables for the IWAENC draft."""

import argparse
import json
import os
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_asn")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache_asn")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPLITS = ["Standard", "Hard", "Worst"]
SPLIT_LABELS = ["SNR 5-20 dB", "SNR 0-8 dB", "SNR <0 dB"]

PRIMARY_MODELS = [
    ("Local", "Local (cIRM)", "#9ECAE1"),
    ("N-Sh", "Graph-F (cIRM)", "#31A354"),
    ("F-Sh", "Graph-All (cIRM)", "#00441B"),
]

ABLATION_MODELS = [
    ("local (IRM)", "Local (IRM)"),
    ("Bottleneck-only feature sharing", "Bottleneck-only (V7)"),
    ("Bottleneck feature sharing", "Bottleneck+FiLM (V7)"),
    ("Input level sharing", "Graph-F (IRM)"),
    ("fully shared network (IRM)", "Graph-All (IRM)"),
    ("selected local (cIRM)", "Local (cIRM)"),
    ("selected Neighbourhood shared network (cIRM)", "Graph-F (cIRM)"),
    ("selected fully shared network (cIRM)", "Graph-All (cIRM)"),
]

METRICS = [
    ("sdr_out", "SI-SDR (dB)", "{:.2f}"),
    ("pesq_out", "PESQ", "{:.2f}"),
    ("stoi_out", "STOI", "{:.3f}"),
]


def safe_name(name):
    return name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")


def load_rows(cache_dir, split, model):
    for fname in (f"{split}_{model}.json", f"{split}_{safe_name(model)}.json"):
        path = os.path.join(cache_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)["rows"]
    return []


def load_data(cache_dir, models):
    return {
        split: {model: load_rows(cache_dir, split, model) for _, model, _ in models}
        for split in SPLITS
    }


def vals(rows, key):
    return np.array([r[key] for r in rows if key in r and r[key] == r[key]], dtype=float)


def mean(rows, key):
    v = vals(rows, key)
    return float(v.mean()) if v.size else float("nan")


def std(rows, key):
    v = vals(rows, key)
    return float(v.std(ddof=0)) if v.size else float("nan")


def pool(data, models, splits, snr_fn=None, rt60_fn=None, spk=None):
    out = defaultdict(list)
    for split in splits:
        for model in models:
            for row in data[split].get(model, []):
                if snr_fn and not snr_fn(row["snr"]):
                    continue
                if rt60_fn and not rt60_fn(row["rt60"]):
                    continue
                if spk is not None and row["spk"] != spk:
                    continue
                out[model].append(row)
    return out


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png")
    plt.close(fig)
    print(f"Saved {base}.pdf/.png")


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
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


def fig_graph_gain(data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    fig.subplots_adjust(wspace=0.30)
    x = np.arange(len(SPLITS))

    local_model = "Local (cIRM)"
    shared = PRIMARY_MODELS[1:]
    for ax, (key, ylabel, fmt) in zip(axes, METRICS):
        for label, model, color in shared:
            gains = []
            for split in SPLITS:
                gains.append(mean(data[split][model], key) - mean(data[split][local_model], key))
            ax.plot(
                x,
                gains,
                marker="o",
                markersize=7,
                linewidth=2.2,
                color=color,
                label=label.replace("\n", " "),
                zorder=3,
            )
            ax.annotate(
                fmt.format(gains[-1]),
                xy=(x[-1], gains[-1]),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=color,
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(SPLIT_LABELS)
        ax.set_ylabel(f"Gain over local: {ylabel}")
        ax.set_title(ylabel, fontweight="bold")

    axes[0].legend(framealpha=0.9, loc="upper left")
    fig.suptitle("Graph Gain vs Acoustic Difficulty (CRM)", fontsize=12, fontweight="bold", y=1.03)
    save(fig, out_dir, "fig2_graph_gain_progression_cirm")


def fig_speaker_counts(data, out_dir):
    models = [m for _, m, _ in PRIMARY_MODELS]
    filters = [{"spk": 1}, {"spk": 2}, {"spk": 3}]
    labels = ["1 speaker", "2 speakers", "3 speakers"]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    fig.subplots_adjust(wspace=0.30)
    x = np.arange(len(labels))
    width = 0.22
    offsets = np.linspace(-width, width, len(PRIMARY_MODELS))

    for ax, (key, ylabel, fmt) in zip(axes, METRICS):
        for i, (display, model, color) in enumerate(PRIMARY_MODELS):
            y = []
            for kwargs in filters:
                rows = pool(data, [model], ["Standard", "Hard"], **kwargs)[model]
                y.append(mean(rows, key))
            ax.bar(x + offsets[i], y, width, color=color, edgecolor="white", linewidth=0.6, label=display.replace("\n", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, framealpha=0.9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Performance by Speaker Count (CRM)", fontsize=12, fontweight="bold", y=1.03)
    fig.subplots_adjust(bottom=0.18)
    save(fig, out_dir, "fig3_breakdown_spk_cirm")


def fig_sisdr_heatmap(data, out_dir):
    rows = [
        ("Standard", lambda d, m: d["Standard"][m]),
        ("Hard", lambda d, m: d["Hard"][m]),
        ("Worst", lambda d, m: d["Worst"][m]),
        ("1 spk", lambda d, m: pool(d, [m], ["Standard", "Hard"], spk=1)[m]),
        ("2 spk", lambda d, m: pool(d, [m], ["Standard", "Hard"], spk=2)[m]),
        ("3 spk", lambda d, m: pool(d, [m], SPLITS, spk=3)[m]),
        ("Easy SNR", lambda d, m: pool(d, [m], SPLITS, snr_fn=lambda v: v >= 10)[m]),
        ("Med SNR", lambda d, m: pool(d, [m], SPLITS, snr_fn=lambda v: 0 <= v < 10)[m]),
        ("Hard SNR", lambda d, m: pool(d, [m], SPLITS, snr_fn=lambda v: v < 0)[m]),
        ("Dry RT60", lambda d, m: pool(d, [m], SPLITS, rt60_fn=lambda v: v < 0.5)[m]),
        ("Mild RT60", lambda d, m: pool(d, [m], SPLITS, rt60_fn=lambda v: 0.5 <= v < 0.7)[m]),
        ("Wet RT60", lambda d, m: pool(d, [m], SPLITS, rt60_fn=lambda v: v >= 0.7)[m]),
    ]

    models = [m for _, m, _ in PRIMARY_MODELS]
    matrix = np.full((len(rows), len(models)), np.nan)
    gain = np.full((len(rows), 2), np.nan)
    for i, (_, fn) in enumerate(rows):
        for j, model in enumerate(models):
            matrix[i, j] = mean(fn(data, model), "sdr_out")
        gain[i, 0] = matrix[i, 1] - matrix[i, 0]
        gain[i, 1] = matrix[i, 2] - matrix[i, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 6.0), gridspec_kw={"width_ratios": [1.25, 0.85]})
    fig.subplots_adjust(wspace=0.08)

    im1 = ax1.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=-1, vmax=15)
    plt.colorbar(im1, ax=ax1, label="Output SI-SDR (dB)", shrink=0.80, pad=0.02)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax1.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8, fontweight="bold")
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([x[0] for x in PRIMARY_MODELS], rotation=20, ha="right", fontsize=8)
    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels([r[0] for r in rows])
    ax1.set_title("Output SI-SDR (CRM)", fontweight="bold")

    im2 = ax2.imshow(gain, cmap="RdYlGn", aspect="auto", vmin=0, vmax=5.5)
    plt.colorbar(im2, ax=ax2, label="Gain over local (dB)", shrink=0.80, pad=0.02)
    for i in range(gain.shape[0]):
        for j in range(gain.shape[1]):
            ax2.text(j, i, f"{gain[i, j]:+.1f}", ha="center", va="center", fontsize=8, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Neighbourhood\nshared", "fully shared"], rotation=20, ha="right", fontsize=8)
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([""] * len(rows))
    ax2.set_title("Graph gain", fontweight="bold")

    fig.suptitle("SI-SDR Results and Graph Gain (CRM)", fontsize=12, fontweight="bold", y=1.01)
    save(fig, out_dir, "fig9_heatmap_sisdr_cirm")


def fig_delta_vs_input(data, out_dir):
    split = "Worst"
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    n_bins = 12

    for label, model, color in PRIMARY_MODELS:
        rows = data[split][model]
        xi = vals(rows, "sdr_in")
        yi = vals(rows, "delta")
        idx = np.argsort(xi)
        xi = xi[idx]
        yi = yi[idx]
        bins = np.array_split(np.arange(len(xi)), n_bins)
        bx = np.array([xi[b].mean() for b in bins])
        by = np.array([yi[b].mean() for b in bins])
        bs = np.array([yi[b].std(ddof=0) for b in bins])
        ax.fill_between(bx, by - bs, by + bs, color=color, alpha=0.12, linewidth=0)
        ax.plot(bx, by, color=color, marker="o", markersize=5, linewidth=2.2, label=label.replace("\n", " "))

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_xlabel("Input SI-SDR (dB)")
    ax.set_ylabel("Enhancement gain: output - input SI-SDR (dB)")
    ax.set_title("Enhancement Gain vs Input Quality (CRM)", fontweight="bold")
    ax.legend(framealpha=0.9)
    save(fig, out_dir, "fig7_delta_vs_input_cirm")


def table_main(data, table_dir):
    os.makedirs(table_dir, exist_ok=True)
    path = os.path.join(table_dir, "table_main_cirm_metrics.tex")
    cols = "ll" + "ccc" * len(SPLITS)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main results for the selected cIRM model family. Values are means over scenes and nodes.}",
        "\\label{tab:main_cirm_metrics}",
        "\\setlength{\\tabcolsep}{3.5pt}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "Model & Metric & \\multicolumn{3}{c}{Standard} & \\multicolumn{3}{c}{Hard} & \\multicolumn{3}{c}{Worst} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
        " & & In & Out & Gain & In & Out & Gain & In & Out & Gain \\\\",
        "\\midrule",
    ]
    for display, model, _ in PRIMARY_MODELS:
        for metric_i, (out_key, metric_name, fmt) in enumerate(METRICS):
            in_key = out_key.replace("_out", "_in")
            cells = []
            for split in SPLITS:
                in_v = mean(data[split][model], in_key)
                out_v = mean(data[split][model], out_key)
                gain_v = out_v - in_v
                cells.extend([fmt.format(in_v), fmt.format(out_v), fmt.format(gain_v)])
            model_cell = display.replace("\n", " ") if metric_i == 0 else ""
            lines.append(f"{model_cell} & {metric_name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


def table_compact_all_metrics(data, table_dir):
    os.makedirs(table_dir, exist_ok=True)
    path = os.path.join(table_dir, "table_main_compact_all_metrics.tex")

    rows = [
        ("Overall", "Standard", lambda m: data["Standard"][m]),
        ("Overall", "Hard", lambda m: data["Hard"][m]),
        ("Overall", "Worst", lambda m: data["Worst"][m]),
        ("Speakers", "1 spk", lambda m: pool(data, [m], ["Standard", "Hard"], spk=1)[m]),
        ("Speakers", "2 spk", lambda m: pool(data, [m], ["Standard", "Hard"], spk=2)[m]),
        ("Speakers", "3 spk", lambda m: pool(data, [m], SPLITS, spk=3)[m]),
        ("SNR", "Easy", lambda m: pool(data, [m], SPLITS, snr_fn=lambda v: v >= 10)[m]),
        ("SNR", "Medium", lambda m: pool(data, [m], SPLITS, snr_fn=lambda v: 0 <= v < 10)[m]),
        ("SNR", "Hard", lambda m: pool(data, [m], SPLITS, snr_fn=lambda v: v < 0)[m]),
        ("RT60", "Dry", lambda m: pool(data, [m], SPLITS, rt60_fn=lambda v: v < 0.5)[m]),
        ("RT60", "Mild", lambda m: pool(data, [m], SPLITS, rt60_fn=lambda v: 0.5 <= v < 0.7)[m]),
        ("RT60", "Wet", lambda m: pool(data, [m], SPLITS, rt60_fn=lambda v: v >= 0.7)[m]),
    ]
    row_spans = {"Overall": 3, "Speakers": 3, "SNR": 3, "RT60": 3}
    models = ["Local (cIRM)", "Graph-F (cIRM)", "Graph-All (cIRM)"]

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Mean SI-SDR, PESQ, and STOI across acoustic conditions for the selected cIRM model family. ``In'' denotes the input mixture. ``Gain'' is F-Sh $-$ Local in SI-SDR.}",
        "\\label{tab:main_cirm_all_metrics}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\renewcommand{\\arraystretch}{0.98}",
        "\\begin{tabular}{llc|ccc|c|c|ccc|c|ccc}",
        "\\hline",
        "\\textbf{Setting} & \\textbf{Condition} & \\multicolumn{5}{c|}{\\textbf{SI-SDR (dB)}} & \\multicolumn{4}{c|}{\\textbf{PESQ}} & \\multicolumn{4}{c}{\\textbf{STOI}} \\\\",
        "& & In & Local & N-Sh & F-Sh & Gain & In & Local & N-Sh & F-Sh & In & Local & N-Sh & F-Sh \\\\",
        "\\hline",
    ]

    prev_setting = None
    for setting, condition, row_fn in rows:
        if prev_setting and setting != prev_setting:
            lines.append("\\hline")
        setting_cell = f"\\multirow{{{row_spans[setting]}}}{{*}}{{{setting}}}" if setting != prev_setting else ""
        row_data = {m: row_fn(m) for m in models}
        local_rows = row_data["Local (cIRM)"]
        sisdr_in = mean(local_rows, "sdr_in")
        pesq_in = mean(local_rows, "pesq_in")
        stoi_in = mean(local_rows, "stoi_in")
        sisdr_out = [mean(row_data[m], "sdr_out") for m in models]
        pesq_out = [mean(row_data[m], "pesq_out") for m in models]
        stoi_out = [mean(row_data[m], "stoi_out") for m in models]
        gain = sisdr_out[2] - sisdr_out[0]
        cells = [
            setting_cell,
            condition,
            f"{sisdr_in:.2f}",
            f"{sisdr_out[0]:.2f}",
            f"{sisdr_out[1]:.2f}",
            f"{sisdr_out[2]:.2f}",
            f"{gain:+.2f}",
            f"{pesq_in:.2f}",
            f"{pesq_out[0]:.2f}",
            f"{pesq_out[1]:.2f}",
            f"{pesq_out[2]:.2f}",
            f"{stoi_in:.3f}",
            f"{stoi_out[0]:.3f}",
            f"{stoi_out[1]:.3f}",
            f"{stoi_out[2]:.3f}",
        ]
        lines.append(" & ".join(cells) + " \\\\")
        prev_setting = setting

    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


def table_ablation(cache_dir, table_dir):
    os.makedirs(table_dir, exist_ok=True)
    path = os.path.join(table_dir, "table_ablation_extra.tex")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation and model-family comparison. Values are output SI-SDR (dB).}",
        "\\label{tab:ablation_extra}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Model & Standard & Hard & Worst \\\\",
        "\\midrule",
    ]
    for display, model in ABLATION_MODELS:
        cells = []
        for split in SPLITS:
            rows = load_rows(cache_dir, split, model)
            cells.append(f"{mean(rows, 'sdr_out'):.2f}")
        lines.append(f"{display} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/home/rrame12/Desktop/Research/ASN/paper_output/per_scene")
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "generated_figures", "final_cirm"))
    parser.add_argument("--table_dir", default=os.path.join(os.path.dirname(__file__), "generated_tables", "final_cirm"))
    args = parser.parse_args()

    configure_style()
    data = load_data(args.cache_dir, PRIMARY_MODELS)
    fig_graph_gain(data, args.out_dir)
    fig_speaker_counts(data, args.out_dir)
    fig_sisdr_heatmap(data, args.out_dir)
    fig_delta_vs_input(data, args.out_dir)
    table_main(data, args.table_dir)
    table_compact_all_metrics(data, args.table_dir)
    table_ablation(args.cache_dir, args.table_dir)


if __name__ == "__main__":
    main()
