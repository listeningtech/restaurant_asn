#!/usr/bin/env python3
"""Standalone script to generate PESQ and STOI LaTeX tables from cached JSON."""
import os, json
from collections import defaultdict

CACHE = "/home/rrame12/Desktop/Research/ASN/paper_output/per_scene"
OUT   = "/home/rrame12/Desktop/Research/ASN/paper_output/tables"

MODEL_NAMES = [
    "Local (IRM)", "Graph-F (IRM)", "Graph-All (IRM)",
    "Local (PSM)", "Graph-F (PSM)", "Graph-All (PSM)",
    "Local (cIRM)", "Graph-F (cIRM)", "Graph-All (cIRM)",
]
SPLITS = ["Standard", "Hard", "Worst"]

def _nanmean(vals):
    v = [x for x in vals if x == x]
    return sum(v)/len(v) if v else float("nan")

def agg(rows):
    if not rows: return None
    n = len(rows)
    return {
        "in":       sum(r["sdr_in"]  for r in rows)/n,
        "out":      sum(r["sdr_out"] for r in rows)/n,
        "pesq_in":  _nanmean([r.get("pesq_in",  float("nan")) for r in rows]),
        "pesq_out": _nanmean([r.get("pesq_out", float("nan")) for r in rows]),
        "stoi_in":  _nanmean([r.get("stoi_in",  float("nan")) for r in rows]),
        "stoi_out": _nanmean([r.get("stoi_out", float("nan")) for r in rows]),
        "n": n,
    }

data = {s: {} for s in SPLITS}
for split in SPLITS:
    for mn in MODEL_NAMES:
        p = os.path.join(CACHE, f"{split}_{mn}.json")
        if os.path.exists(p):
            with open(p) as f:
                data[split][mn] = json.load(f)["rows"]
        else:
            print(f"MISSING: {p}")

def pool(mn, splits=None):
    rows = []
    for s in (splits or SPLITS):
        rows += data[s].get(mn, [])
    return rows

def gbnum(snr_fn=None, rt60_fn=None, spk_val=None, splits=None):
    def fn(m):
        rows = pool(m, splits)
        return [r for r in rows if
                (snr_fn  is None or snr_fn(r["snr"]))  and
                (rt60_fn is None or rt60_fn(r["rt60"])) and
                (spk_val is None or r["spk"] == spk_val)]
    return fn

table_rows = [
    ("Overall",  "Standard",              lambda m: data["Standard"].get(m,[])),
    ("",         "Hard",                  lambda m: data["Hard"].get(m,[])),
    ("",         "Worst",                 lambda m: data["Worst"].get(m,[])),
    ("Speaker",  "1 speaker",             gbnum(spk_val=1, splits=["Standard","Hard"])),
    ("",         "2 speakers",            gbnum(spk_val=2, splits=["Standard","Hard"])),
    ("",         "3 speakers",            gbnum(spk_val=3)),
    ("SNR",      r"Easy ($\geq$10~dB)",   gbnum(snr_fn=lambda v: v>=10)),
    ("",         r"Medium (0--10~dB)",    gbnum(snr_fn=lambda v: 0<=v<10)),
    ("",         r"Hard ($<$0~dB)",       gbnum(snr_fn=lambda v: v<0)),
    ("RT60",     r"Dry ($<$0.5~s)",       gbnum(rt60_fn=lambda v: v<0.5)),
    ("",         r"Mild (0.5--0.7~s)",    gbnum(rt60_fn=lambda v: 0.5<=v<0.7)),
    ("",         r"Wet ($\geq$0.7~s)",    gbnum(rt60_fn=lambda v: v>=0.7)),
]

def emit_table(metric_out, mix_in, caption, label, fmt_fn, path):
    col_spec = "ll c " + " ".join(["c"]*len(MODEL_NAMES))
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
        for mn in MODEL_NAMES:
            a = agg(rows_fn(mn))
            if a:
                v = a.get(mix_in)
                if v is not None and v == v:
                    mix_val = v; break

        out_vals = []
        for mn in MODEL_NAMES:
            a = agg(rows_fn(mn))
            v = a.get(metric_out) if a else None
            out_vals.append(v if (v is not None and v == v) else None)

        valid = [v for v in out_vals if v is not None]
        best  = max(valid) if valid else None

        mix_str   = fmt_fn(mix_val)
        cell_strs = [fmt_fn(v, bold=(v is not None and best is not None and abs(v-best)<0.005))
                     for v in out_vals]
        L.append((group if group else "") + " & " + condition
                 + " & " + mix_str + " & " + " & ".join(cell_strs) + r" \\")

    L.append(r"\bottomrule\end{tabular}\end{table*}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L)+"\n")
    print(f"Saved {path}")

def fmt2(v, bold=False):
    if v is None: return "--"
    s = f"{v:.2f}"
    return r"\textbf{" + s + "}" if bold else s

def fmt3(v, bold=False):
    if v is None: return "--"
    s = f"{v:.3f}"
    return r"\textbf{" + s + "}" if bold else s

emit_table("pesq_out","pesq_in",
    r"Mean output PESQ (WB-MOS, 1--4.5, higher is better). \emph{Mixture} = input PESQ. Best per row \textbf{bold}.",
    "tab:results_pesq", fmt2,
    os.path.join(OUT, "table_results_pesq.tex"))

emit_table("stoi_out","stoi_in",
    r"Mean output STOI (0--1, higher is better). \emph{Mixture} = input STOI. Best per row \textbf{bold}.",
    "tab:results_stoi", fmt3,
    os.path.join(OUT, "table_results_stoi.tex"))

for split_label, split_key in [("Worst", "Worst"), ("Standard", "Standard")]:
    print(f"\n=== PESQ + STOI ({split_label} split) ===")
    print(f"{'Model':<22} {'PESQ-in':>8} {'PESQ-out':>9} {'STOI-in':>8} {'STOI-out':>9}")
    print("-"*62)
    for mn in MODEL_NAMES:
        a = agg(data[split_key].get(mn, []))
        if not a: print(f"  {mn:<20}  N/A"); continue
        print(f"  {mn:<20} {a['pesq_in']:>8.3f} {a['pesq_out']:>9.3f} {a['stoi_in']:>8.4f} {a['stoi_out']:>9.4f}")
