#!/usr/bin/env python3
"""
eval_all_models_table.py

Evaluate all trained models (V8, V8 ablations, V9, V10) on a given dataset
split and print a combined comparison table with breakdowns by:
  - Overall
  - Local speaker count (1 / 2 / 3)
  - SNR tier (hard / medium / easy)
  - RT60 tier (dry / mid / wet)

Results are also saved to --out_dir as JSON.

Usage (standard test split):
  python eval_all_models_table.py \
    --data /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk \
    --split test \
    --out_dir /home/rrame12/Desktop/Research/ASN/eval_table_standard

Usage (hard test split):
  python eval_all_models_table.py \
    --data /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_hard \
    --split test \
    --out_dir /home/rrame12/Desktop/Research/ASN/eval_table_hard
"""

import os, sys, glob, json, argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

SR  = 16000
EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model imports (avoid loading all at startup)
# ─────────────────────────────────────────────────────────────────────────────

def load_v8(run_dir, ckpt_path, device):
    from train_graph_input_crossnode_v8 import GraphInputCrossNodeV8, compute_si_sdr
    a = _read_json(os.path.join(run_dir, "args.json"))
    model = GraphInputCrossNodeV8(
        n_mics        = int(a["resolved_n_mics"]),
        n_fft         = int(a.get("nfft", 512)),
        hop           = int(a.get("hop", 128)),
        base          = int(a.get("base", 32)),
        depth         = int(a.get("depth", 4)),
        drop          = 0.0,
        graph_dim     = int(a.get("graph_dim", 128)),
        graph_heads   = int(a.get("graph_heads", 4)),
        graph_dropout = 0.0,
        use_graph     = bool(a.get("graph_enabled", not a.get("disable_graph", False))),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    return model.to(device).eval(), compute_si_sdr


def load_v8_ablation(run_dir, ckpt_path, device):
    from train_graph_ablations_v8 import GraphAblationV8, compute_si_sdr
    a = _read_json(os.path.join(run_dir, "args.json"))
    model = GraphAblationV8(
        n_mics        = int(a["resolved_n_mics"]),
        n_fft         = int(a.get("nfft", 512)),
        hop           = int(a.get("hop", 128)),
        base          = int(a.get("base", 32)),
        depth         = int(a.get("depth", 4)),
        drop          = 0.0,
        graph_dim     = int(a.get("graph_dim", 128)),
        graph_heads   = int(a.get("graph_heads", 4)),
        graph_dropout = 0.0,
        use_graph     = bool(a.get("graph_enabled", not a.get("disable_graph", False))),
        adj_mode      = str(a.get("adj_mode", "fixed")),
        inject_level1 = bool(a.get("inject_level1", False)),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    return model.to(device).eval(), compute_si_sdr


def load_v9(run_dir, ckpt_path, device):
    from train_graph_psm_v9 import GraphPSMV9, compute_si_sdr
    a = _read_json(os.path.join(run_dir, "args.json"))
    model = GraphPSMV9(
        n_mics        = int(a["resolved_n_mics"]),
        n_fft         = int(a.get("nfft", 1024)),
        hop           = int(a.get("hop", 256)),
        base          = int(a.get("base", 32)),
        depth         = int(a.get("depth", 4)),
        drop          = 0.0,
        graph_dim     = int(a.get("graph_dim", 128)),
        graph_heads   = int(a.get("graph_heads", 4)),
        graph_dropout = 0.0,
        use_graph     = bool(a.get("graph_enabled", not a.get("disable_graph", False))),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    return model.to(device).eval(), compute_si_sdr


def load_v10(run_dir, ckpt_path, device):
    from train_graph_cirm_v10 import GraphCIRMV10, compute_si_sdr
    a = _read_json(os.path.join(run_dir, "args.json"))
    model = GraphCIRMV10(
        n_mics        = int(a["resolved_n_mics"]),
        n_fft         = int(a.get("nfft", 1024)),
        hop           = int(a.get("hop", 256)),
        base          = int(a.get("base", 32)),
        depth         = int(a.get("depth", 4)),
        drop          = 0.0,
        graph_dim     = int(a.get("graph_dim", 128)),
        graph_heads   = int(a.get("graph_heads", 4)),
        graph_dropout = 0.0,
        use_graph     = bool(a.get("graph_enabled", not a.get("disable_graph", False))),
        mask_scale    = float(a.get("mask_scale", 10.0)),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    return model.to(device).eval(), compute_si_sdr


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_json(p):
    with open(p) as f: return json.load(f)

def _write_json(p, o):
    with open(p, "w") as f: json.dump(o, f, indent=2)

def _ensure(p): os.makedirs(p, exist_ok=True)

def snr_tier(v):
    if v <  0: return "worst  (SNR<0dB)"
    if v <  5: return "hard   (SNR 0-5dB)"
    if v < 10: return "medium (SNR 5-10dB)"
    return            "easy   (SNR>=10dB)"

def rt60_tier(v):
    if v < 0.5:  return "dry  (RT60<0.5s)"
    if v < 0.65: return "mid  (RT60 0.5-0.65s)"
    if v < 0.80: return "wet  (RT60 0.65-0.8s)"
    return              "very_wet (RT60>=0.8s)"


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

@torch.no_grad()
def evaluate_model(model, compute_si_sdr, loader, device, meta):
    rows = []
    for batch in tqdm(loader, desc="  eval", ncols=100, leave=False):
        Y = batch["Y"].to(device); S = batch["S"].to(device); A = batch["A"].to(device)
        Yhat      = model(Y, A)
        sdr_out   = compute_si_sdr(Yhat,          S)   # (B,K)
        sdr_in    = compute_si_sdr(Y[:,:,0,:],    S)   # (B,K)
        B, K, _   = Yhat.shape
        for b in range(B):
            m = meta.get(batch["sid"][b], batch["li"][b])
            snr  = float(m["snr_db"]) if m else float("nan")
            rt60 = float(m["rt60"])   if m else float("nan")
            spt  = m["speakers_per_table"] if m else [None]*K
            for k in range(K):
                lspk = spt[k] if k < len(spt) else None
                rows.append({
                    "sdr_in":  float(sdr_in[b,k].item()),
                    "sdr_out": float(sdr_out[b,k].item()),
                    "snr":     snr, "rt60": rt60,
                    "g_spk":   f"{lspk} spk" if lspk is not None else "?",
                    "g_snr":   snr_tier(snr)  if m else "?",
                    "g_rt60":  rt60_tier(rt60) if m else "?",
                })
    return rows


def agg(rows):
    n = len(rows)
    if n == 0: return {"in": float("nan"), "out": float("nan"), "delta": float("nan"), "n": 0}
    si = sum(r["sdr_in"]  for r in rows) / n
    so = sum(r["sdr_out"] for r in rows) / n
    return {"in": si, "out": so, "delta": so - si, "n": n}


def breakdown(rows, key):
    buckets = defaultdict(list)
    for r in rows: buckets[r[key]].append(r)
    return {g: agg(v) for g, v in sorted(buckets.items())}


# ─────────────────────────────────────────────────────────────────────────────
# Table printing
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title, all_results, key, width=28):
    """Print one breakdown section across all models."""
    # Collect all group labels
    labels = sorted({g for r in all_results.values() for g in r["by"][key]})
    col = 10

    print(f"\n{'─'*120}")
    print(f"  {title}")
    print(f"  {'Group':<{width}}  " + "  ".join(
        f"{'Δ '+n:>{col}}" for n in all_results))
    print(f"  {'─'*width}  " + "  ".join(f"{'─'*col}" for _ in all_results))
    for g in labels:
        row = f"  {g:<{width}}  "
        for stats in all_results.values():
            v = stats["by"][key].get(g)
            if v:
                row += f"  {v['delta']:>+{col-2}.2f}dB"
            else:
                row += f"  {'  n/a':>{col}}"
        print(row)


def print_overall(all_results):
    col = 10
    print(f"\n{'═'*120}")
    print(f"  {'Model':<38}  {'Graph':>6}  {'Mask':>5}  {'SI-SDR_in':>{col}}  {'SI-SDR_out':>{col}}  {'Delta':>{col}}  {'N':>6}")
    print(f"  {'─'*38}  {'─'*6}  {'─'*5}  {'─'*col}  {'─'*col}  {'─'*col}  {'─'*6}")
    for name, r in all_results.items():
        v = r["overall"]
        print(f"  {name:<38}  {r['graph']:>6}  {r['mask']:>5}  "
              f"{v['in']:>{col}.3f}  {v['out']:>{col}.3f}  {v['delta']:>+{col}.3f}  {v['n']:>6}")
    print(f"  {'═'*38}  {'═'*6}  {'═'*5}  {'═'*col}  {'═'*col}  {'═'*col}  {'═'*6}")


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

ASN_ROOT = os.path.join(os.path.dirname(__file__), "..")

def model_registry():
    """Returns list of (display_name, graph_label, mask_label, run_dir, loader_fn)."""
    r = ASN_ROOT
    return [
        # name                           graph   mask   run_dir                                           loader
        ("V8 OFF  (IRM, fixed)",         "OFF",  "IRM", f"{r}/runs_input_crossnode_v8_off",               load_v8),
        ("V8 ON   (IRM, fixed)",         "ON",   "IRM", f"{r}/runs_input_crossnode_v8_on",                load_v8),
        ("V8 ON   (IRM, fixed, ext)",    "ON",   "IRM", f"{r}/runs_input_crossnode_v8_on_ext",            load_v8),
        ("V8 ON   (IRM, alltoall)",      "ON",   "IRM", f"{r}/runs_ablation_v8_alltoall",                 load_v8_ablation),
        ("V8 ON   (IRM, fixed+L1inj)",   "ON",   "IRM", f"{r}/runs_ablation_v8_multilevel",               load_v8_ablation),
        ("V9 OFF  (PSM, fixed)",         "OFF",  "PSM", f"{r}/runs_psm_v9_off",                          load_v9),
        ("V9 ON   (PSM, fixed)",         "ON",   "PSM", f"{r}/runs_psm_v9_on",                           load_v9),
        ("V10 OFF (cIRM, fixed)",        "OFF",  "cIRM",f"{r}/runs_cirm_v10_off",                        load_v10),
        ("V10 ON  (cIRM, fixed)",        "ON",   "cIRM",f"{r}/runs_cirm_v10_on",                         load_v10),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",        required=True,  help="Dataset root (with manifest.json)")
    ap.add_argument("--split",       default="test", choices=["train","val","test"])
    ap.add_argument("--out_dir",     required=True,  help="Where to save per-model JSON results")
    ap.add_argument("--batch",       type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--models",      nargs="*", default=None,
                    help="Subset of model names to run (default: all)")
    args = ap.parse_args()

    _ensure(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split_dir = os.path.join(args.data, args.split)
    meta      = ShardMeta(split_dir)
    ds        = Dataset(split_dir)
    loader    = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0))

    print(f"Dataset: {args.data}  [{args.split}]  {len(ds)} scenes")

    registry = model_registry()
    if args.models:
        registry = [(n,g,m,d,lf) for (n,g,m,d,lf) in registry
                    if any(k.lower() in n.lower() for k in args.models)]

    all_results = {}
    cache_path  = os.path.join(args.out_dir, "all_results.json")

    # Load cached results if available (skip re-eval)
    cached = {}
    if os.path.exists(cache_path):
        cached = _read_json(cache_path)
        print(f"Loaded {len(cached)} cached results from {cache_path}")

    for name, graph_lbl, mask_lbl, run_dir, loader_fn in registry:
        safe_name = name.replace(" ", "_").replace("(","").replace(")","").replace(",","").replace("/","_")

        if name in cached:
            print(f"[cached] {name}")
            all_results[name] = cached[name]
            continue

        ckpt = os.path.join(run_dir, "best.pt")
        if not os.path.exists(ckpt):
            print(f"[SKIP] {name} — no checkpoint at {ckpt}")
            continue
        if not os.path.exists(os.path.join(run_dir, "args.json")):
            print(f"[SKIP] {name} — no args.json")
            continue

        print(f"\n[{name}]")
        model, compute_si_sdr = loader_fn(run_dir, ckpt, device)
        rows = evaluate_model(model, compute_si_sdr, loader, device, meta)
        del model; torch.cuda.empty_cache()

        result = {
            "name":    name,
            "graph":   graph_lbl,
            "mask":    mask_lbl,
            "run_dir": run_dir,
            "overall": agg(rows),
            "by": {
                "spk":  breakdown(rows, "g_spk"),
                "snr":  breakdown(rows, "g_snr"),
                "rt60": breakdown(rows, "g_rt60"),
            },
        }
        all_results[name] = result
        # Save incrementally
        _write_json(cache_path, all_results)
        _write_json(os.path.join(args.out_dir, f"{safe_name}.json"), result)

    # ── Print tables ──────────────────────────────────────────────────────────
    if not all_results:
        print("No results to display."); return

    print(f"\n\n{'═'*120}")
    print(f"  RESULTS  |  data={args.data}  split={args.split}")
    print(f"{'═'*120}")

    print_overall(all_results)
    print_section("Delta SI-SDR by local speaker count", all_results, "spk",  width=22)
    print_section("Delta SI-SDR by SNR tier",            all_results, "snr",  width=22)
    print_section("Delta SI-SDR by RT60 tier",           all_results, "rt60", width=22)

    # ── Graph-gain summary ────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("  Graph gain (ON – OFF, matched mask type)")
    print(f"  {'Model pair':<45}  {'Overall Δ':>10}  {'Hard SNR Δ':>12}")
    pairs = [
        ("V8 ON   (IRM, fixed)",       "V8 OFF  (IRM, fixed)"),
        ("V8 ON   (IRM, fixed, ext)",  "V8 OFF  (IRM, fixed)"),
        ("V8 ON   (IRM, alltoall)",    "V8 OFF  (IRM, fixed)"),
        ("V8 ON   (IRM, fixed+L1inj)", "V8 OFF  (IRM, fixed)"),
        ("V9 ON   (PSM, fixed)",       "V9 OFF  (PSM, fixed)"),
        ("V10 ON  (cIRM, fixed)",      "V10 OFF (cIRM, fixed)"),
    ]
    for on_name, off_name in pairs:
        if on_name not in all_results or off_name not in all_results: continue
        on  = all_results[on_name]
        off = all_results[off_name]
        gain_overall = on["overall"]["out"] - off["overall"]["out"]

        # hardest SNR bucket
        on_snr  = on["by"]["snr"]
        off_snr = off["by"]["snr"]
        hard_keys = [k for k in on_snr if "hard" in k.lower()]
        if hard_keys and hard_keys[0] in off_snr:
            hk = hard_keys[0]
            gain_hard = on_snr[hk]["out"] - off_snr[hk]["out"]
            hard_str  = f"{gain_hard:>+10.2f} dB"
        else:
            hard_str = "       n/a"
        label = f"{on_name.split('(')[0].strip()} vs {off_name.split('(')[0].strip()}"
        print(f"  {label:<45}  {gain_overall:>+8.2f} dB  {hard_str}")

    _write_json(cache_path, all_results)
    print(f"\n  Full results saved → {args.out_dir}/")


if __name__ == "__main__":
    main()
