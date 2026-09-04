#!/usr/bin/env python3
"""
generate_hard_test_split.py

Generate a configurable test-only split of the RestaurantSim_v4 dataset
with controllable SNR / RT60 / speaker count ranges.

Presets:
  hard   : SNR  0–8 dB,  RT60 0.5–0.8s,  2–3 spk/table
  worst  : SNR -8–0 dB,  RT60 0.6–0.9s,  3–3 spk/table

Usage:
  # hard split
  python generate_hard_test_split.py \
    --out /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_hard \
    --n_test 500 --shard_size 50

  # worst-case split
  python generate_hard_test_split.py \
    --out /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_worst \
    --snr_min -8 --snr_max 0 --rt60_min 0.6 --rt60_max 0.9 \
    --min_speakers_per_table 3 --max_speakers_per_table 3 \
    --seed_base 199000000 --n_test 500 --shard_size 50
"""

import os
import sys
import json
import argparse

DATA_ROOT = "/home/rrame12/Desktop/Datasets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",        default=os.path.join(DATA_ROOT, "RestaurantSim_v4_m1_hard"))
    ap.add_argument("--libri",      default=os.path.join(DATA_ROOT, "LibriSpeech"))
    ap.add_argument("--vctk",       default=os.path.join(DATA_ROOT, "VCTK"))
    ap.add_argument("--musan",      default=os.path.join(DATA_ROOT, "musan"))
    ap.add_argument("--n_test",     type=int,   default=500)
    ap.add_argument("--shard_size", type=int,   default=50)
    ap.add_argument("--seed_base",  type=int,   default=99_000_000)
    ap.add_argument("--snr_min",    type=float, default=0.0)
    ap.add_argument("--snr_max",    type=float, default=8.0)
    ap.add_argument("--rt60_min",   type=float, default=0.50)
    ap.add_argument("--rt60_max",   type=float, default=0.80)
    ap.add_argument("--min_speakers_per_table", type=int, default=2)
    ap.add_argument("--max_speakers_per_table", type=int, default=3)
    args = ap.parse_args()

    # ── patch module-level constants BEFORE importing the generator ──────────
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../V3"))
    import generate_restaurant_dataset_varspk as gen
    gen.RT60_RANGE   = (args.rt60_min, args.rt60_max)
    gen.SNR_DB_RANGE = (args.snr_min,  args.snr_max)
    # ─────────────────────────────────────────────────────────────────────────

    os.makedirs(args.out, exist_ok=True)

    print(f"Test split: {args.out}")
    print(f"  RT60  : {gen.RT60_RANGE}")
    print(f"  SNR   : {gen.SNR_DB_RANGE} dB")
    print(f"  Spk/table: [{args.min_speakers_per_table}, {args.max_speakers_per_table}]")
    print(f"  n_test: {args.n_test}  seed_base: {args.seed_base}")
    print(f"  out   : {args.out}")

    idx = gen.build_index(args.libri, args.vctk, args.musan)
    print(f"Speech: {len(idx.speech_paths)}  Noise: {len(idx.noise_paths)}  Music: {len(idx.music_paths)}")

    gen.generate_split(
        idx=idx,
        out_dir=args.out,
        split="test",
        n_scenes=args.n_test,
        shard_size=args.shard_size,
        seed_base=args.seed_base,
        mics_per_table=1,
        mic_radius=gen.MIC_RADIUS,
        min_speakers_per_table=args.min_speakers_per_table,
        max_speakers_per_table=args.max_speakers_per_table,
    )

    manifest = {
        "sr": gen.SR,
        "duration_s": gen.DURATION_S,
        "T": gen.T,
        "K": gen.K_TABLES,
        "M": 1,
        "P_mode": "variable_per_table",
        "P_min": args.min_speakers_per_table,
        "P_max": args.max_speakers_per_table,
        "room_dims": gen.ROOM_DIMS,
        "rt60_range": list(gen.RT60_RANGE),
        "snr_db_range": list(gen.SNR_DB_RANGE),
        "counts": {"test": args.n_test},
        "shard_size": args.shard_size,
        "seed_base": args.seed_base,
        "notes": f"Test-only split. SNR {args.snr_min}–{args.snr_max} dB, RT60 {args.rt60_min}–{args.rt60_max}s, {args.min_speakers_per_table}–{args.max_speakers_per_table} spk/table.",
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Test split written to {args.out}/test/")


if __name__ == "__main__":
    main()
