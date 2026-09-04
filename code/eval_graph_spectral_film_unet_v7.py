#!/usr/bin/env python3
"""
eval_graph_spectral_film_unet_v7.py

Evaluate GraphSpectralFiLMUNetV7 on the test (or val) split.

Checkpoint format: plain state_dict saved by train_graph_spectral_film_unet_v7.py
Architecture args are read from args.json in the same run directory.

Outputs (written to --out):
  metrics_summary.json   — single dict with SI-SDR_in, SI-SDR_out, delta
  metrics_per_scene.json — one row per (batch_idx, sample, table)
  audio_examples/        — optional wavs (--save_scenes N)

Example (graph ON):
  CUDA_VISIBLE_DEVICES=0 python eval_graph_spectral_film_unet_v7.py \\
    --run_dir /home/rrame12/Desktop/Research/ASN/runs_spectral_film_v7_on \\
    --data    /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk \\
    --out     /home/rrame12/Desktop/Research/ASN/runs_spectral_film_v7_on/eval_test \\
    --split test --batch 4 --save_scenes 6

Example (graph OFF):
  CUDA_VISIBLE_DEVICES=0 python eval_graph_spectral_film_unet_v7.py \\
    --run_dir /home/rrame12/Desktop/Research/ASN/runs_spectral_film_v7_off \\
    --data    /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk \\
    --out     /home/rrame12/Desktop/Research/ASN/runs_spectral_film_v7_off/eval_test \\
    --split test --batch 4

You can also pass --ckpt explicitly to override the default best.pt in run_dir.
"""

import os
import glob
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from tqdm import tqdm

# Import model and helpers from the training script so definitions stay in sync.
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_graph_spectral_film_unet_v7 import (
    GraphSpectralFiLMUNetV7,
    compute_si_sdr,
)

SR  = 16000
EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)

def write_json(path: str, obj: Dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (identical to training script, kept self-contained for easy use)
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
                B = d["Y"].shape[0]
            for i in range(B):
                self.index.append((sid, i))

        self._cache_store: Dict[int, Dict] = {}

    def __len__(self):
        return len(self.index)

    def _load(self, sid: int) -> Dict:
        if self.cache and sid in self._cache_store:
            return self._cache_store[sid]
        d = np.load(self.shard_paths[sid])
        sh = {k: d[k] for k in d.files}
        d.close()
        if self.cache:
            self._cache_store[sid] = sh
        return sh

    def __getitem__(self, idx: int):
        sid, li = self.index[idx]
        sh = self._load(sid)
        Y  = sh["Y"][li]                            # (K, M, T)
        S  = sh["target_refclean"][li]              # (K, T)
        A  = sh["adj"][li].astype(np.float32)       # (K, K)
        _, _, TT = Y.shape
        if self.crop_samples is not None and self.crop_samples < TT:
            start = np.random.randint(0, TT - self.crop_samples + 1)
            Y = Y[:, :, start:start + self.crop_samples]
            S = S[:, start:start + self.crop_samples]
        return {
            "Y": torch.from_numpy(Y).float(),
            "S": torch.from_numpy(S).float(),
            "A": torch.from_numpy(A).float(),
        }


def collate_fn(batch):
    return {
        "Y": torch.stack([b["Y"] for b in batch], 0),
        "S": torch.stack([b["S"] for b in batch], 0),
        "A": torch.stack([b["A"] for b in batch], 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model restore
# ─────────────────────────────────────────────────────────────────────────────

def restore_model(ckpt_path: str, run_dir: str,
                  n_mics_override: Optional[int] = None) -> Tuple[nn.Module, Dict]:
    """
    Load args.json from run_dir, rebuild the model, load state_dict from ckpt_path.
    Returns (model_on_cpu, args_dict).
    """
    args_path = os.path.join(run_dir, "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in run_dir: {run_dir}")
    a = read_json(args_path)

    n_mics = n_mics_override if n_mics_override is not None else int(a["resolved_n_mics"])

    model = GraphSpectralFiLMUNetV7(
        n_mics        = n_mics,
        n_fft         = int(a.get("nfft", 512)),
        hop           = int(a.get("hop", 128)),
        base          = int(a.get("base", 32)),
        depth         = int(a.get("depth", 4)),
        drop          = 0.0,                         # always 0 at eval
        graph_dim     = int(a.get("graph_dim", 128)),
        graph_heads   = int(a.get("graph_heads", 4)),
        graph_dropout = 0.0,                         # always 0 at eval
        use_graph     = bool(a.get("graph_enabled", not a.get("disable_graph", False))),
        multiscale    = bool(a.get("multiscale", False)),
    )

    state = torch.load(ckpt_path, map_location="cpu")
    # state_dict saved directly (not wrapped in a dict)
    model.load_state_dict(state, strict=True)
    return model, a


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device,
             out_dir: str, save_scenes: int = 0) -> Dict:
    model.eval()
    ensure_dir(out_dir)
    audio_dir = os.path.join(out_dir, "audio_examples")
    if save_scenes > 0:
        ensure_dir(audio_dir)

    total_in = total_out = total_n = 0.0
    saved    = 0
    per_scene: List[Dict] = []

    pbar = tqdm(loader, desc="Evaluating", ncols=110)
    for batch_idx, batch in enumerate(pbar):
        Y = batch["Y"].to(device, non_blocking=True)   # (B, K, M, T)
        S = batch["S"].to(device, non_blocking=True)   # (B, K, T)
        A = batch["A"].to(device, non_blocking=True)   # (B, K, K)

        Yhat = model(Y, A)                             # (B, K, T)

        sisdr_out = compute_si_sdr(Yhat,         S)    # (B, K)
        sisdr_in  = compute_si_sdr(Y[:, :, 0, :], S)  # (B, K)
        delta     = sisdr_out - sisdr_in

        B, K, _ = Yhat.shape
        total_in  += sisdr_in.sum().item()
        total_out += sisdr_out.sum().item()
        total_n   += B * K

        pbar.set_postfix({
            "SI-SDR_in":  f"{total_in  / total_n:.2f}",
            "SI-SDR_out": f"{total_out / total_n:.2f}",
            "delta":      f"{(total_out - total_in) / total_n:+.2f}",
        })

        for b in range(B):
            for k in range(K):
                per_scene.append({
                    "batch_index":    int(batch_idx),
                    "sample_in_batch": int(b),
                    "table":          int(k),
                    "sisdr_in_db":    float(sisdr_in[b, k].item()),
                    "sisdr_out_db":   float(sisdr_out[b, k].item()),
                    "delta_db":       float(delta[b, k].item()),
                })

        # Save audio examples
        if saved < save_scenes:
            n_save   = min(B, save_scenes - saved)
            Y_np     = Y.cpu().numpy()
            S_np     = S.cpu().numpy()
            Yhat_np  = Yhat.cpu().numpy()
            for b in range(n_save):
                scene_dir = os.path.join(audio_dir, f"scene_{saved:03d}")
                ensure_dir(scene_dir)
                for k in range(K):
                    sf.write(os.path.join(scene_dir, f"table{k}_mix_ref.wav"),
                             Y_np[b, k, 0], SR)
                    sf.write(os.path.join(scene_dir, f"table{k}_target.wav"),
                             S_np[b, k], SR)
                    sf.write(os.path.join(scene_dir, f"table{k}_enh.wav"),
                             Yhat_np[b, k], SR)
                saved += 1
                if saved >= save_scenes:
                    break

    n       = max(total_n, 1)
    summary = {
        "sisdr_in_db":  total_in  / n,
        "sisdr_out_db": total_out / n,
        "delta_db":     (total_out - total_in) / n,
        "n_examples":   int(total_n),
    }
    write_json(os.path.join(out_dir, "metrics_summary.json"), summary)
    write_json(os.path.join(out_dir, "metrics_per_scene.json"), {"rows": per_scene})
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir",    required=True,
                    help="run directory containing args.json (and best.pt by default)")
    ap.add_argument("--data",       required=True,
                    help="dataset root (must contain the chosen --split sub-directory)")
    ap.add_argument("--out",        required=True,
                    help="directory to write eval results")
    ap.add_argument("--ckpt",       default=None,
                    help="explicit checkpoint path; defaults to <run_dir>/best.pt")
    ap.add_argument("--split",      default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch",      type=int,   default=4)
    ap.add_argument("--num_workers", type=int,  default=6)
    ap.add_argument("--full_length", type=int,  default=1,
                    help="1 = evaluate on full-length clips (recommended); 0 = use crop from args.json")
    ap.add_argument("--save_scenes", type=int,  default=0,
                    help="number of scenes to save as wav files")
    ap.add_argument("--n_mics_override", type=int, default=None)
    args = ap.parse_args()

    ensure_dir(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt if args.ckpt else os.path.join(args.run_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, train_args = restore_model(ckpt_path, args.run_dir,
                                      n_mics_override=args.n_mics_override)
    model = model.to(device)

    # Crop / full-length
    if args.full_length:
        crop_samples = None
    else:
        crop_s = float(train_args.get("crop_s", 2.0))
        crop_samples = None if crop_s <= 0 else int(round(crop_s * SR))

    split_dir = os.path.join(args.data, args.split)
    ds     = RestaurantSceneDataset(split_dir, crop_samples=crop_samples, cache=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    graph_on = bool(train_args.get("graph_enabled",
                    not train_args.get("disable_graph", False)))
    print("=" * 70)
    print("Evaluate  GraphSpectralFiLMUNetV7")
    print(f"  device      : {device}")
    print(f"  run_dir     : {args.run_dir}")
    print(f"  ckpt        : {ckpt_path}")
    print(f"  data        : {args.data}  [{args.split}]")
    print(f"  out         : {args.out}")
    print(f"  graph       : {'ON' if graph_on else 'OFF'}")
    print(f"  multiscale  : {bool(train_args.get('multiscale', False))}")
    print(f"  n_mics      : {train_args.get('resolved_n_mics')}")
    print(f"  crop_samples: {crop_samples}")
    print(f"  n_scenes    : {len(ds)}")
    print("=" * 70)

    summary = evaluate(model, loader, device, args.out, save_scenes=args.save_scenes)

    print()
    print("=== RESULTS ===")
    print(f"  mode        : {'GRAPH_ON' if graph_on else 'GRAPH_OFF'}")
    print(f"  SI-SDR_in   : {summary['sisdr_in_db']:.3f} dB")
    print(f"  SI-SDR_out  : {summary['sisdr_out_db']:.3f} dB")
    print(f"  Delta       : {summary['delta_db']:+.3f} dB")
    print(f"  N examples  : {summary['n_examples']}")
    print(f"  results     : {args.out}/metrics_summary.json")


if __name__ == "__main__":
    main()
