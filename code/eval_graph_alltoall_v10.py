#!/usr/bin/env python3
"""
eval_graph_cirm_v10.py

Evaluate GraphCIRMV10 on the test (or val) split.

Reports overall SI-SDR plus breakdowns by:
  - local speaker count  (speakers_per_table for this node: 1, 2, 3, ...)
  - difficulty tier      (SNR-based: easy / medium / hard)
  - RT60 tier            (dry / mid / reverberant)

Metadata is read from the shard_XXXX.jsonl sidecar files, which contain
per-scene fields: speakers_per_table, snr_db, rt60.

Example (graph ON):
  CUDA_VISIBLE_DEVICES=0 python eval_graph_cirm_v10.py \\
    --run_dir /home/rrame12/Desktop/Research/ASN/runs_cirm_v10_on \\
    --data    /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk \\
    --out     /home/rrame12/Desktop/Research/ASN/runs_cirm_v10_on/eval_test \\
    --split test --batch 4 --save_scenes 6

Example (graph OFF):
  CUDA_VISIBLE_DEVICES=0 python eval_graph_cirm_v10.py \\
    --run_dir /home/rrame12/Desktop/Research/ASN/runs_cirm_v10_off \\
    --data    /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk \\
    --out     /home/rrame12/Desktop/Research/ASN/runs_cirm_v10_off/eval_test \\
    --split test --batch 4
"""

import os
import glob
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_graph_alltoall_v10 import (
    GraphAllToAllV10,
    compute_si_sdr,
)

SR  = 16000
EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_json(path):
    with open(path) as f: return json.load(f)

def write_json(path, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# JSONL sidecar loader
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class ShardMeta:
    """
    Loads and caches the .jsonl sidecar that accompanies each .npz shard.
    Provides per-(shard, local_idx) lookup for scene-level metadata:
      - speakers_per_table: List[int]  (length K)
      - snr_db: float
      - rt60:   float
    """
    def __init__(self, split_dir: str):
        self.split_dir = split_dir
        self._cache: Dict[int, List[Dict]] = {}

    def _load(self, sid: int) -> List[Dict]:
        if sid in self._cache:
            return self._cache[sid]
        jsonl_path = os.path.join(
            self.split_dir, f"shard_{sid:04d}.jsonl")
        if not os.path.exists(jsonl_path):
            return []
        records = load_jsonl(jsonl_path)
        self._cache[sid] = records
        return records

    def get(self, sid: int, local_idx: int) -> Optional[Dict]:
        records = self._load(sid)
        if not records or local_idx >= len(records):
            return None
        return records[local_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — now also returns shard/local indices for metadata lookup
# ─────────────────────────────────────────────────────────────────────────────

class RestaurantSceneDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir: str, crop_samples: Optional[int], cache: bool = True):
        super().__init__()
        self.crop_samples = crop_samples
        self.cache        = cache
        self.shard_paths  = sorted(glob.glob(os.path.join(split_dir, "shard_*.npz")))
        if not self.shard_paths:
            raise RuntimeError(f"No shards found in {split_dir}")
        self.index: List[Tuple[int, int]] = []
        for sid, sp in enumerate(self.shard_paths):
            with np.load(sp) as d:
                for i in range(d["Y"].shape[0]):
                    self.index.append((sid, i))
        self._cache_store: Dict[int, Dict] = {}

    def __len__(self): return len(self.index)

    def _load(self, sid):
        if self.cache and sid in self._cache_store:
            return self._cache_store[sid]
        d  = np.load(self.shard_paths[sid])
        sh = {k: d[k] for k in d.files}; d.close()
        if self.cache: self._cache_store[sid] = sh
        return sh

    def __getitem__(self, idx):
        sid, li = self.index[idx]
        sh = self._load(sid)
        Y  = sh["Y"][li]; S = sh["target_refclean"][li]
        A  = sh["adj"][li].astype(np.float32)
        _, _, TT = Y.shape
        if self.crop_samples is not None and self.crop_samples < TT:
            start = np.random.randint(0, TT - self.crop_samples + 1)
            Y = Y[:, :, start:start + self.crop_samples]
            S = S[:, start:start + self.crop_samples]
        return {
            "Y":   torch.from_numpy(Y).float(),
            "S":   torch.from_numpy(S).float(),
            "A":   torch.from_numpy(A).float(),
            "sid": sid,
            "li":  li,
        }


def collate_fn(batch):
    return {
        "Y":   torch.stack([b["Y"]   for b in batch], 0),
        "S":   torch.stack([b["S"]   for b in batch], 0),
        "A":   torch.stack([b["A"]   for b in batch], 0),
        "sid": [b["sid"] for b in batch],
        "li":  [b["li"]  for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model restore
# ─────────────────────────────────────────────────────────────────────────────

def restore_model(ckpt_path: str, run_dir: str,
                  n_mics_override: Optional[int] = None) -> Tuple[nn.Module, Dict]:
    a      = read_json(os.path.join(run_dir, "args.json"))
    n_mics = n_mics_override if n_mics_override is not None else int(a["resolved_n_mics"])
    graph_enabled = bool(a.get("graph_enabled", not a.get("disable_graph", False)))
    model  = GraphAllToAllV10(
        n_mics        = n_mics,
        n_fft         = int(a.get("nfft", 1024)),
        hop           = int(a.get("hop", 256)),
        base          = int(a.get("base", 32)),
        depth         = int(a.get("depth", 4)),
        drop          = 0.0,
        graph_dim     = int(a.get("graph_dim", 128)),
        graph_heads   = int(a.get("graph_heads", 4)),
        graph_dropout = 0.0,
        use_graph     = graph_enabled,
        mask_scale    = float(a.get("mask_scale", 10.0)),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    return model, a


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty tiers
# ─────────────────────────────────────────────────────────────────────────────

def snr_tier(snr_db: float) -> str:
    if snr_db < 10.0:  return "hard   (SNR< 10dB)"
    if snr_db < 15.0:  return "medium (SNR 10-15dB)"
    return               "easy   (SNR>=15dB)"

def rt60_tier(rt60: float) -> str:
    if rt60 < 0.4:  return "dry  (RT60<0.4s)"
    if rt60 < 0.5:  return "mid  (RT60 0.4-0.5s)"
    return           "wet  (RT60>=0.5s)"


# ─────────────────────────────────────────────────────────────────────────────
# Grouped-stats helper
# ─────────────────────────────────────────────────────────────────────────────

def group_stats(rows: List[Dict], key: str) -> Dict[str, Dict]:
    """
    Given a list of per-node dicts each with a 'group_{key}' field,
    returns {group_label: {sisdr_in, sisdr_out, delta, n}}.
    """
    buckets: Dict[str, Dict] = defaultdict(lambda: {"sum_in": 0.0, "sum_out": 0.0, "n": 0})
    for r in rows:
        g = r.get(f"group_{key}", "unknown")
        buckets[g]["sum_in"]  += r["sisdr_in_db"]
        buckets[g]["sum_out"] += r["sisdr_out_db"]
        buckets[g]["n"]       += 1
    result = {}
    for g, v in sorted(buckets.items()):
        n = max(v["n"], 1)
        result[g] = {
            "sisdr_in_db":  v["sum_in"]  / n,
            "sisdr_out_db": v["sum_out"] / n,
            "delta_db":     (v["sum_out"] - v["sum_in"]) / n,
            "n":            v["n"],
        }
    return result


def print_breakdown(title: str, stats: Dict[str, Dict]):
    print(f"\n  {title}")
    print(f"  {'Group':<28}  {'SI-SDR_in':>10}  {'SI-SDR_out':>10}  {'Delta':>8}  {'N':>6}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}")
    for label, v in stats.items():
        print(f"  {label:<28}  {v['sisdr_in_db']:>10.3f}  {v['sisdr_out_db']:>10.3f}  "
              f"{v['delta_db']:>+8.3f}  {v['n']:>6}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device,
             out_dir: str, shard_meta: ShardMeta,
             save_scenes: int = 0) -> Tuple[Dict, List[Dict]]:
    model.eval()
    ensure_dir(out_dir)
    audio_dir = os.path.join(out_dir, "audio_examples")
    if save_scenes > 0: ensure_dir(audio_dir)

    total_in = total_out = total_n = 0.0
    saved    = 0
    per_scene: List[Dict] = []

    pbar = tqdm(loader, desc="Evaluating", ncols=110)
    for batch in pbar:
        Y    = batch["Y"].to(device, non_blocking=True)
        S    = batch["S"].to(device, non_blocking=True)
        A    = batch["A"].to(device, non_blocking=True)
        sids = batch["sid"]   # list of ints, length B
        lis  = batch["li"]    # list of ints, length B

        Yhat      = model(Y, A)
        sisdr_out = compute_si_sdr(Yhat,          S)   # (B, K)
        sisdr_in  = compute_si_sdr(Y[:, :, 0, :], S)   # (B, K)
        delta     = sisdr_out - sisdr_in
        B, K, _   = Yhat.shape

        total_in  += sisdr_in.sum().item()
        total_out += sisdr_out.sum().item()
        total_n   += B * K

        pbar.set_postfix({
            "in":    f"{total_in  / total_n:.2f}",
            "out":   f"{total_out / total_n:.2f}",
            "delta": f"{(total_out - total_in) / total_n:+.2f}",
        })

        for b in range(B):
            meta = shard_meta.get(sids[b], lis[b])
            # Scene-level metadata (same for all K nodes in this scene)
            snr  = float(meta.get("snr_db", meta.get("snr", float("nan")))) if meta else float("nan")
            rt60 = float(meta.get("rt60", float("nan"))) if meta else float("nan")
            spk_per_table = meta["speakers_per_table"] if meta else [None] * K

            for k in range(K):
                local_spk = spk_per_table[k] if k < len(spk_per_table) else None
                per_scene.append({
                    "sid":           sids[b],
                    "li":            lis[b],
                    "table":         int(k),
                    "sisdr_in_db":   float(sisdr_in[b, k].item()),
                    "sisdr_out_db":  float(sisdr_out[b, k].item()),
                    "delta_db":      float(delta[b, k].item()),
                    "snr_db":        snr,
                    "rt60":          rt60,
                    "local_speakers": int(local_spk) if local_spk is not None else -1,
                    # group keys for breakdown
                    "group_spk":     f"{local_spk} local spk" if local_spk is not None else "unknown",
                    "group_snr":     snr_tier(snr)  if meta else "unknown",
                    "group_rt60":    rt60_tier(rt60) if meta else "unknown",
                })

        # Save audio examples
        if saved < save_scenes:
            n_save  = min(B, save_scenes - saved)
            Y_np    = Y.cpu().numpy()
            S_np    = S.cpu().numpy()
            Yh_np   = Yhat.cpu().numpy()
            for b in range(n_save):
                sd = os.path.join(audio_dir, f"scene_{saved:03d}"); ensure_dir(sd)
                for k in range(K):
                    sf.write(os.path.join(sd, f"table{k}_mix_ref.wav"), Y_np[b, k, 0], SR)
                    sf.write(os.path.join(sd, f"table{k}_target.wav"),  S_np[b, k],    SR)
                    sf.write(os.path.join(sd, f"table{k}_enh.wav"),     Yh_np[b, k],   SR)
                saved += 1
                if saved >= save_scenes: break

    n       = max(total_n, 1)
    summary = {
        "sisdr_in_db":  total_in  / n,
        "sisdr_out_db": total_out / n,
        "delta_db":     (total_out - total_in) / n,
        "n_examples":   int(total_n),
    }
    return summary, per_scene


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir",     required=True)
    ap.add_argument("--data",        required=True)
    ap.add_argument("--out",         required=True)
    ap.add_argument("--ckpt",        default=None)
    ap.add_argument("--split",       default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch",       type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--full_length", type=int, default=1,
                    help="1 = full-length clips (recommended); 0 = use crop from args.json")
    ap.add_argument("--save_scenes", type=int, default=0)
    ap.add_argument("--n_mics_override", type=int, default=None)
    args = ap.parse_args()

    ensure_dir(args.out)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt or os.path.join(args.run_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, train_args = restore_model(ckpt_path, args.run_dir, args.n_mics_override)
    model = model.to(device)
    graph_on = bool(train_args.get("graph_enabled",
                    not train_args.get("disable_graph", False)))

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
        pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    print("=" * 70)
    print("Evaluate  GraphAllToAllV10")
    print(f"  device  : {device}")
    print(f"  run_dir : {args.run_dir}")
    print(f"  ckpt    : {ckpt_path}")
    print(f"  data    : {args.data}  [{args.split}]")
    print(f"  graph   : {'ON' if graph_on else 'OFF'}")
    print(f"  n_mics  : {train_args.get('resolved_n_mics')}")
    print(f"  scenes  : {len(ds)}")
    print("=" * 70)

    summary, per_scene = evaluate(
        model, loader, device, args.out,
        shard_meta=shard_meta, save_scenes=args.save_scenes)

    # ── Breakdowns ──────────────────────────────────────────────────────────
    spk_stats  = group_stats(per_scene, "spk")
    snr_stats  = group_stats(per_scene, "snr")
    rt60_stats = group_stats(per_scene, "rt60")

    # ── Print ────────────────────────────────────────────────────────────────
    mode_str = "GRAPH_ON" if graph_on else "GRAPH_OFF"
    print(f"\n{'='*70}")
    print(f"  RESULTS  [{mode_str}]")
    print(f"{'='*70}")
    print(f"  Overall SI-SDR_in  : {summary['sisdr_in_db']:.3f} dB")
    print(f"  Overall SI-SDR_out : {summary['sisdr_out_db']:.3f} dB")
    print(f"  Overall Delta      : {summary['delta_db']:+.3f} dB")
    print(f"  N node-examples    : {summary['n_examples']}")

    print_breakdown("By local speaker count", spk_stats)
    print_breakdown("By SNR difficulty",      snr_stats)
    print_breakdown("By RT60 (reverb)",       rt60_stats)

    # ── Save ─────────────────────────────────────────────────────────────────
    write_json(os.path.join(args.out, "metrics_summary.json"), {
        **summary,
        "by_local_speakers": spk_stats,
        "by_snr_tier":       snr_stats,
        "by_rt60_tier":      rt60_stats,
    })
    write_json(os.path.join(args.out, "metrics_per_scene.json"), {"rows": per_scene})
    print(f"\n  Saved → {args.out}/metrics_summary.json")


if __name__ == "__main__":
    main()
