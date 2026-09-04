#!/usr/bin/env python3
"""
eval_graph_ablations_v8.py

Evaluate GraphAblationV8 checkpoints (train_graph_ablations_v8.py).
Reads adj_mode and inject_level1 from args.json automatically.

Example:
  CUDA_VISIBLE_DEVICES=0 python eval_graph_ablations_v8.py \\
    --run_dir /home/rrame12/Desktop/Research/ASN/runs_ablation_v8_alltoall \\
    --data    /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk \\
    --out     /home/rrame12/Desktop/Research/ASN/runs_ablation_v8_alltoall/eval_test \\
    --split test --batch 4 --save_scenes 4
"""

import os, glob, json, argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_graph_ablations_v8 import GraphAblationV8, compute_si_sdr

SR = 16000

def read_json(p):
    with open(p) as f: return json.load(f)
def write_json(p, o):
    with open(p,"w") as f: json.dump(o, f, indent=2)
def ensure_dir(p): os.makedirs(p, exist_ok=True)


# ── Dataset ──────────────────────────────────────────────────────────────────

class RestaurantSceneDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir, crop_samples, cache=True):
        super().__init__()
        self.crop_samples = crop_samples; self.cache = cache
        self.shard_paths  = sorted(glob.glob(os.path.join(split_dir, "shard_*.npz")))
        if not self.shard_paths: raise RuntimeError(f"No shards in {split_dir}")
        self.index = []
        for sid, sp in enumerate(self.shard_paths):
            with np.load(sp) as d:
                for i in range(d["Y"].shape[0]): self.index.append((sid, i))
        self._store = {}
    def __len__(self): return len(self.index)
    def _load(self, sid):
        if self.cache and sid in self._store: return self._store[sid]
        d = np.load(self.shard_paths[sid]); sh = {k:d[k] for k in d.files}; d.close()
        if self.cache: self._store[sid] = sh
        return sh
    def __getitem__(self, idx):
        sid, li = self.index[idx]; sh = self._load(sid)
        Y = sh["Y"][li]; S = sh["target_refclean"][li]; A = sh["adj"][li].astype(np.float32)
        _, _, TT = Y.shape
        if self.crop_samples is not None and self.crop_samples < TT:
            s = np.random.randint(0, TT - self.crop_samples + 1)
            Y = Y[:,:,s:s+self.crop_samples]; S = S[:,s:s+self.crop_samples]
        return {"Y": torch.from_numpy(Y).float(), "S": torch.from_numpy(S).float(),
                "A": torch.from_numpy(A).float(), "sid": sid, "li": li}

def collate_fn(batch):
    return {"Y": torch.stack([b["Y"] for b in batch],0),
            "S": torch.stack([b["S"] for b in batch],0),
            "A": torch.stack([b["A"] for b in batch],0),
            "sid": [b["sid"] for b in batch], "li": [b["li"] for b in batch]}


# ── Metadata sidecar ──────────────────────────────────────────────────────────

class ShardMeta:
    def __init__(self, split_dir):
        self.split_dir = split_dir; self._cache = {}
    def _load(self, sid):
        if sid in self._cache: return self._cache[sid]
        p = os.path.join(self.split_dir, f"shard_{sid:04d}.jsonl")
        if not os.path.exists(p): return []
        with open(p) as f: recs = [json.loads(l) for l in f if l.strip()]
        self._cache[sid] = recs; return recs
    def get(self, sid, li):
        recs = self._load(sid)
        return recs[li] if recs and li < len(recs) else None

def snr_tier(v):
    if v < 10:  return "hard   (SNR< 10dB)"
    if v < 15:  return "medium (SNR 10-15dB)"
    return              "easy   (SNR>=15dB)"

def rt60_tier(v):
    if v < 0.4: return "dry  (RT60<0.4s)"
    if v < 0.5: return "mid  (RT60 0.4-0.5s)"
    return             "wet  (RT60>=0.5s)"

def group_stats(rows, key):
    b = defaultdict(lambda: {"sum_in":0.0,"sum_out":0.0,"n":0})
    for r in rows:
        g = r.get(f"group_{key}", "unknown")
        b[g]["sum_in"] += r["sisdr_in_db"]; b[g]["sum_out"] += r["sisdr_out_db"]; b[g]["n"] += 1
    return {g: {"sisdr_in_db": v["sum_in"]/max(v["n"],1),
                "sisdr_out_db": v["sum_out"]/max(v["n"],1),
                "delta_db": (v["sum_out"]-v["sum_in"])/max(v["n"],1), "n": v["n"]}
            for g, v in sorted(b.items())}

def print_breakdown(title, stats):
    print(f"\n  {title}")
    print(f"  {'Group':<28}  {'SI-SDR_in':>10}  {'SI-SDR_out':>10}  {'Delta':>8}  {'N':>6}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}")
    for label, v in stats.items():
        print(f"  {label:<28}  {v['sisdr_in_db']:>10.3f}  {v['sisdr_out_db']:>10.3f}"
              f"  {v['delta_db']:>+8.3f}  {v['n']:>6}")


# ── Model restore ─────────────────────────────────────────────────────────────

def restore_model(ckpt_path, run_dir, n_mics_override=None):
    a = read_json(os.path.join(run_dir, "args.json"))
    n_mics = n_mics_override if n_mics_override is not None else int(a["resolved_n_mics"])
    graph_enabled = bool(a.get("graph_enabled", not a.get("disable_graph", False)))
    model = GraphAblationV8(
        n_mics        = n_mics,
        n_fft         = int(a.get("nfft", 512)),
        hop           = int(a.get("hop", 128)),
        base          = int(a.get("base", 32)),
        depth         = int(a.get("depth", 4)),
        drop          = 0.0,
        graph_dim     = int(a.get("graph_dim", 128)),
        graph_heads   = int(a.get("graph_heads", 4)),
        graph_dropout = 0.0,
        use_graph     = graph_enabled,
        adj_mode      = str(a.get("adj_mode", "fixed")),
        inject_level1 = bool(a.get("inject_level1", False)),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    return model, a


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, out_dir, shard_meta, save_scenes=0):
    model.eval(); ensure_dir(out_dir)
    audio_dir = os.path.join(out_dir, "audio_examples")
    if save_scenes > 0: ensure_dir(audio_dir)

    total_in = total_out = total_n = 0.0; saved = 0; per_scene = []
    pbar = tqdm(loader, desc="Evaluating", ncols=110)
    for batch in pbar:
        Y = batch["Y"].to(device, non_blocking=True)
        S = batch["S"].to(device, non_blocking=True)
        A = batch["A"].to(device, non_blocking=True)
        Yhat      = model(Y, A)
        sisdr_out = compute_si_sdr(Yhat, S)
        sisdr_in  = compute_si_sdr(Y[:,:,0,:], S)
        delta     = sisdr_out - sisdr_in
        B, K, _   = Yhat.shape
        total_in += sisdr_in.sum().item(); total_out += sisdr_out.sum().item(); total_n += B*K
        pbar.set_postfix({"in":  f"{total_in/total_n:.2f}",
                          "out": f"{total_out/total_n:.2f}",
                          "delta": f"{(total_out-total_in)/total_n:+.2f}"})
        for b in range(B):
            meta = shard_meta.get(batch["sid"][b], batch["li"][b])
            snr  = float(meta["snr_db"]) if meta else float("nan")
            rt60 = float(meta["rt60"])   if meta else float("nan")
            spk_per_table = meta["speakers_per_table"] if meta else [None]*K
            for k in range(K):
                lspk = spk_per_table[k] if k < len(spk_per_table) else None
                per_scene.append({
                    "sid": batch["sid"][b], "li": batch["li"][b], "table": k,
                    "sisdr_in_db":   float(sisdr_in[b,k].item()),
                    "sisdr_out_db":  float(sisdr_out[b,k].item()),
                    "delta_db":      float(delta[b,k].item()),
                    "snr_db": snr, "rt60": rt60,
                    "local_speakers": int(lspk) if lspk is not None else -1,
                    "group_spk":  f"{lspk} local spk" if lspk is not None else "unknown",
                    "group_snr":  snr_tier(snr)  if meta else "unknown",
                    "group_rt60": rt60_tier(rt60) if meta else "unknown",
                })
        if saved < save_scenes:
            n_sv = min(B, save_scenes-saved)
            Yn = Y.cpu().numpy(); Sn = S.cpu().numpy(); Yhn = Yhat.cpu().numpy()
            for b in range(n_sv):
                sd = os.path.join(audio_dir, f"scene_{saved:03d}"); ensure_dir(sd)
                for k in range(K):
                    sf.write(os.path.join(sd, f"table{k}_mix_ref.wav"), Yn[b,k,0], SR)
                    sf.write(os.path.join(sd, f"table{k}_target.wav"),  Sn[b,k],   SR)
                    sf.write(os.path.join(sd, f"table{k}_enh.wav"),     Yhn[b,k],  SR)
                saved += 1
                if saved >= save_scenes: break

    n = max(total_n, 1)
    summary = {"sisdr_in_db": total_in/n, "sisdr_out_db": total_out/n,
               "delta_db": (total_out-total_in)/n, "n_examples": int(total_n)}
    return summary, per_scene


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir",      required=True)
    ap.add_argument("--data",         required=True)
    ap.add_argument("--out",          required=True)
    ap.add_argument("--ckpt",         default=None)
    ap.add_argument("--split",        default="test", choices=["train","val","test"])
    ap.add_argument("--batch",        type=int, default=4)
    ap.add_argument("--num_workers",  type=int, default=6)
    ap.add_argument("--full_length",  type=int, default=1)
    ap.add_argument("--save_scenes",  type=int, default=0)
    ap.add_argument("--n_mics_override", type=int, default=None)
    args = ap.parse_args()

    ensure_dir(args.out)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt or os.path.join(args.run_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, train_args = restore_model(ckpt_path, args.run_dir, args.n_mics_override)
    model = model.to(device)
    graph_on  = bool(train_args.get("graph_enabled", not train_args.get("disable_graph", False)))
    adj_mode  = str(train_args.get("adj_mode", "fixed"))
    inj_l1    = bool(train_args.get("inject_level1", False))

    crop_samples = None
    if not args.full_length:
        crop_s = float(train_args.get("crop_s", 2.0))
        crop_samples = None if crop_s <= 0 else int(round(crop_s * SR))

    split_dir  = os.path.join(args.data, args.split)
    shard_meta = ShardMeta(split_dir)
    ds     = RestaurantSceneDataset(split_dir, crop_samples=crop_samples, cache=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False, persistent_workers=(args.num_workers>0))

    print("="*70)
    print("Evaluate  GraphAblationV8")
    print(f"  device        : {device}")
    print(f"  run_dir       : {args.run_dir}")
    print(f"  ckpt          : {ckpt_path}")
    print(f"  graph         : {'ON' if graph_on else 'OFF'}")
    print(f"  adj_mode      : {adj_mode}")
    print(f"  inject_level1 : {inj_l1}")
    print(f"  data [{args.split}] : {len(ds)} scenes")
    print("="*70)

    summary, per_scene = evaluate(model, loader, device, args.out,
                                  shard_meta=shard_meta, save_scenes=args.save_scenes)

    spk_stats  = group_stats(per_scene, "spk")
    snr_stats  = group_stats(per_scene, "snr")
    rt60_stats = group_stats(per_scene, "rt60")

    mode_tag = f"GRAPH_{'ON' if graph_on else 'OFF'} | adj={adj_mode} | inject_l1={inj_l1}"
    print(f"\n{'='*70}")
    print(f"  RESULTS  [{mode_tag}]")
    print(f"{'='*70}")
    print(f"  Overall SI-SDR_in  : {summary['sisdr_in_db']:.3f} dB")
    print(f"  Overall SI-SDR_out : {summary['sisdr_out_db']:.3f} dB")
    print(f"  Overall Delta      : {summary['delta_db']:+.3f} dB")
    print(f"  N node-examples    : {summary['n_examples']}")
    print_breakdown("By local speaker count", spk_stats)
    print_breakdown("By SNR difficulty",      snr_stats)
    print_breakdown("By RT60 (reverb)",       rt60_stats)

    write_json(os.path.join(args.out, "metrics_summary.json"), {
        **summary, "by_local_speakers": spk_stats,
        "by_snr_tier": snr_stats, "by_rt60_tier": rt60_stats})
    write_json(os.path.join(args.out, "metrics_per_scene.json"), {"rows": per_scene})
    print(f"\n  Saved → {args.out}/metrics_summary.json")

if __name__ == "__main__":
    main()
