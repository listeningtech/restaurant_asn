#!/usr/bin/env python3
"""
generate_restaurant_dataset_v6_graph_required.py

RestaurantSim_v6_graph_required
-------------------------------
A harder restaurant / acoustic sensor network dataset designed so that
cross-node information is materially more useful than in earlier versions.

Main design goals vs v5:
1) Default is 1 mic per table.
2) Tables are usually in coupled layouts.
3) Speakers per table vary from 1 to 3.
4) Hard-scene rejection sampling is stronger.
5) Local mic corruption is enabled and can be biased toward hard scenes.
6) Optional partial target masking/dropout at the local mic to create missing
   local evidence that neighboring tables can help disambiguate.
7) Adjacency is distance-based and always keeps at least one neighbor.
8) Rich per-scene JSONL metadata is saved so evaluation can group results by
   speaker count, hardness, corruption, masking, layout, and table-level stats.

Saved shard (.npz) keys:
  Y               : (B,K,M,T) noisy mixtures
  target_refclean : (B,K,T) local-only reverberant clean target at ref mic
  adj             : (B,K,K) adjacency matrix
  rt60            : (B,)
  snr_db          : (B,)
  seed            : (B,)

Saved shard (.jsonl): one JSON object per scene with fields including:
  speakers_per_table
  local_energy_frac
  bleed_sir_db
  is_hard_table
  local_mic_corruption
  local_target_masking
  layout_name
  table_centers
  source metadata

Example:
  python generate_restaurant_dataset_v6_graph_required.py \
    --out /home/rrame12/Desktop/Datasets/RestaurantSim_v6_graph_required \
    --n_train 5000 --n_val 500 --n_test 500 --shard_size 50 \
    --mics_per_table 1 \
    --min_speakers_per_table 1 --max_speakers_per_table 3 \
    --hard_scene_prob 0.9 \
    --min_hard_tables 2 \
    --enable_local_mic_corruption \
    --mic_corrupt_prob 0.35 \
    --enable_local_target_masking \
    --target_mask_prob 0.25
"""

import os
import glob
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import librosa
import pyroomacoustics as pra
from scipy.signal import fftconvolve
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
SR = 16000
DURATION_S = 5.0
T = int(SR * DURATION_S)

K_TABLES = 4
DEFAULT_MICS_PER_TABLE = 1
TABLE_Z = 1.2
MIC_RADIUS = 0.08
SPK_RADIUS_RANGE = (0.30, 0.85)

RT60_RANGE = (0.20, 0.90)
SNR_DB_RANGE = (0.0, 18.0)
MAX_ORDER_CAP = 12

DATA_ROOT_DEFAULT = "/home/rrame12/Desktop/Datasets"
LIBRI_DEFAULT = os.path.join(DATA_ROOT_DEFAULT, "LibriSpeech")
VCTK_DEFAULT = os.path.join(DATA_ROOT_DEFAULT, "VCTK")
MUSAN_DEFAULT = os.path.join(DATA_ROOT_DEFAULT, "musan")

# Hardness defaults
DEFAULT_HARD_LOCAL_FRAC_MAX = 0.72
DEFAULT_HARD_BLEED_SIR_MAX_DB = 5.0
DEFAULT_MIN_HARD_TABLES = 2

# Layout templates intentionally biased toward coupled / line-like interaction.
LAYOUT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "coupled_tight": {
        "room_dims": (8.4, 6.8, 3.0),
        "table_centers": [(2.1, 2.0), (4.5, 2.1), (2.4, 4.6), (4.8, 4.7)],
        "adj_thresh_m": 3.2,
        "weight": 0.42,
    },
    "line_tight": {
        "room_dims": (10.5, 6.2, 3.0),
        "table_centers": [(2.0, 3.0), (4.3, 3.2), (6.6, 3.0), (8.9, 3.1)],
        "adj_thresh_m": 2.8,
        "weight": 0.26,
    },
    "diamond": {
        "room_dims": (9.2, 8.0, 3.0),
        "table_centers": [(4.6, 1.9), (2.3, 4.0), (6.8, 4.0), (4.6, 6.0)],
        "adj_thresh_m": 3.7,
        "weight": 0.18,
    },
    "mid": {
        "room_dims": (10.2, 8.5, 3.0),
        "table_centers": [(2.4, 2.2), (7.7, 2.3), (2.5, 6.3), (7.6, 6.2)],
        "adj_thresh_m": 4.3,
        "weight": 0.10,
    },
    "far": {
        "room_dims": (12.2, 10.2, 3.0),
        "table_centers": [(2.7, 2.6), (9.5, 2.7), (2.8, 7.7), (9.4, 7.5)],
        "adj_thresh_m": 5.0,
        "weight": 0.04,
    },
}

LAYOUT_TEMPLATES_K6: Dict[str, Dict[str, Any]] = {
    "k6_coupled_grid": {
        "room_dims": (12.0, 8.2, 3.0),
        "table_centers": [(2.2, 2.2), (4.7, 2.1), (7.2, 2.2), (2.3, 5.7), (4.8, 5.8), (7.3, 5.7)],
        "adj_thresh_m": 3.1,
        "weight": 0.40,
    },
    "k6_line_tight": {
        "room_dims": (15.2, 6.4, 3.0),
        "table_centers": [(2.0, 3.0), (4.4, 3.1), (6.8, 3.0), (9.2, 3.1), (11.6, 3.0), (14.0, 3.1)],
        "adj_thresh_m": 2.9,
        "weight": 0.25,
    },
    "k6_two_clusters": {
        "room_dims": (13.0, 9.0, 3.0),
        "table_centers": [(2.2, 2.4), (4.5, 2.6), (3.2, 4.8), (8.4, 4.2), (10.7, 4.4), (9.6, 6.6)],
        "adj_thresh_m": 3.0,
        "weight": 0.20,
    },
    "k6_ring": {
        "room_dims": (12.2, 10.0, 3.0),
        "table_centers": [(6.1, 2.0), (9.0, 3.4), (9.0, 6.7), (6.1, 8.1), (3.2, 6.7), (3.2, 3.4)],
        "adj_thresh_m": 3.5,
        "weight": 0.15,
    },
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def list_files_recursive(root: str, exts: Tuple[str, ...]) -> List[str]:
    out: List[str] = []
    for ext in exts:
        out.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    return sorted(out)


def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + eps))


def energy(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sum(np.square(x, dtype=np.float64)) + eps)


def normalize_rms(x: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    r = rms(x)
    if r < 1e-10:
        return x.astype(np.float32, copy=False)
    return (x * (target_rms / r)).astype(np.float32, copy=False)


def random_crop_or_pad_1d(x: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if len(x) < length:
        return np.pad(x, (0, length - len(x)))
    if len(x) == length:
        return x
    start = rng.randint(0, len(x) - length)
    return x[start:start + length]


def crop_or_pad_2d(X: np.ndarray, length: int) -> np.ndarray:
    C, L = X.shape
    if L == length:
        return X
    if L > length:
        return X[:, :length]
    pad = np.zeros((C, length - L), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin_to_db(x: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(x, eps))


def soft_clip(x: np.ndarray, drive: float = 2.0) -> np.ndarray:
    return np.tanh(drive * x).astype(np.float32)


def load_audio_16k(path: str, length: int, rng: random.Random) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = random_crop_or_pad_1d(y, length, rng)
    y = normalize_rms(y, 0.05)
    return y.astype(np.float32)


def choose_noise_track(idx: "CorpusIndex", rng: random.Random) -> str:
    pool: List[str] = []
    if idx.noise_paths:
        pool.extend(idx.noise_paths)
    if idx.music_paths:
        pool.extend(idx.music_paths)
    if not pool:
        pool = idx.babble_paths
    return rng.choice(pool)


def scale_to_target_snr(clean_ref: np.ndarray, noise_ref: np.ndarray, snr_db: float) -> float:
    rc = rms(clean_ref)
    rn = rms(noise_ref)
    if rn < 1e-10:
        return 1.0
    desired_rn = rc / db_to_lin(snr_db)
    return float(desired_rn / rn)


# -----------------------------------------------------------------------------
# Corpus indexing
# -----------------------------------------------------------------------------
@dataclass
class CorpusIndex:
    speech_paths: List[str]
    noise_paths: List[str]
    music_paths: List[str]
    babble_paths: List[str]


def build_index(libri_root: str, vctk_root: str, musan_root: str) -> CorpusIndex:
    libri_paths = list_files_recursive(libri_root, (".flac", ".wav"))
    vctk_paths = list_files_recursive(os.path.join(vctk_root, "wavs"), (".wav",))
    speech_paths = sorted(libri_paths + vctk_paths)
    if len(speech_paths) == 0:
        raise RuntimeError(f"No speech files found under:\n  {libri_root}\n  {os.path.join(vctk_root, 'wavs')}")

    noise_paths = list_files_recursive(os.path.join(musan_root, "noise"), (".wav",))
    music_paths = list_files_recursive(os.path.join(musan_root, "music"), (".wav",))
    babble_paths = list_files_recursive(os.path.join(musan_root, "speech"), (".wav",))
    if len(noise_paths) == 0 and len(music_paths) == 0 and len(babble_paths) == 0:
        raise RuntimeError(f"No MUSAN wavs found under:\n  {musan_root}")

    return CorpusIndex(
        speech_paths=speech_paths,
        noise_paths=noise_paths,
        music_paths=music_paths,
        babble_paths=babble_paths,
    )


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def make_mic_positions(center_xy: Tuple[float, float], m: int, radius: float, z: float) -> np.ndarray:
    cx, cy = center_xy
    if m == 1:
        return np.asarray([[cx], [cy], [z]], dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    zs = np.ones(m, dtype=np.float32) * z
    return np.vstack([xs, ys, zs]).astype(np.float32)


def sample_speaker_position(center_xy: Tuple[float, float], r_range: Tuple[float, float], z: float, rng: random.Random) -> List[float]:
    cx, cy = center_xy
    ang = rng.uniform(0, 2 * np.pi)
    rr = rng.uniform(r_range[0], r_range[1])
    return [cx + rr * np.cos(ang), cy + rr * np.sin(ang), z]


def pairwise_table_distances(table_centers: List[Tuple[float, float]]) -> np.ndarray:
    K = len(table_centers)
    D = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        xi, yi = table_centers[i]
        for j in range(K):
            xj, yj = table_centers[j]
            D[i, j] = float(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
    return D


def make_adjacency(table_centers: List[Tuple[float, float]], thresh_m: float) -> np.ndarray:
    K = len(table_centers)
    A = np.zeros((K, K), dtype=np.float32)
    D = pairwise_table_distances(table_centers)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            A[i, j] = 1.0 if D[i, j] <= thresh_m else 0.0
    for i in range(K):
        if A[i].sum() <= 0:
            candidates = [(D[i, j], j) for j in range(K) if j != i]
            _, jmin = min(candidates, key=lambda x: x[0])
            A[i, jmin] = 1.0
            A[jmin, i] = 1.0
    return A


def jitter_layout(centers: List[Tuple[float, float]], room_dims: Tuple[float, float, float], rng: random.Random, jitter_xy: float) -> List[Tuple[float, float]]:
    Lx, Ly, _ = room_dims
    margin = 0.9
    out: List[Tuple[float, float]] = []
    for x, y in centers:
        x2 = np.clip(x + rng.uniform(-jitter_xy, jitter_xy), margin, Lx - margin)
        y2 = np.clip(y + rng.uniform(-jitter_xy, jitter_xy), margin, Ly - margin)
        out.append((float(x2), float(y2)))
    return out


def choose_layout_template(rng: random.Random, hard_scene_prob: float) -> str:
    names = list(LAYOUT_TEMPLATES.keys())
    base_weights = np.asarray([LAYOUT_TEMPLATES[n]["weight"] for n in names], dtype=np.float64)
    base_weights /= base_weights.sum()

    # Bias hard scenes even more toward tightly coupled layouts.
    if rng.random() < hard_scene_prob:
        boost = {n: w for n, w in zip(names, base_weights)}
        if K_TABLES == 4:
            boost.update({
                "coupled_tight": 0.50,
                "line_tight": 0.27,
                "diamond": 0.15,
                "mid": 0.06,
                "far": 0.02,
            })
        elif K_TABLES == 6:
            boost.update({
                "k6_coupled_grid": 0.42,
                "k6_line_tight": 0.26,
                "k6_two_clusters": 0.20,
                "k6_ring": 0.12,
            })
        weights = np.asarray([boost[n] for n in names], dtype=np.float64)
        weights /= weights.sum()
    else:
        weights = base_weights
    return rng.choices(names, weights=weights.tolist(), k=1)[0]


# -----------------------------------------------------------------------------
# Room simulation helpers
# -----------------------------------------------------------------------------
def build_room(room_dims: Tuple[float, float, float], rt60: float) -> pra.ShoeBox:
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)
    room = pra.ShoeBox(
        room_dims,
        fs=SR,
        materials=pra.Material(e_absorption),
        max_order=min(MAX_ORDER_CAP, max_order),
    )
    return room


def add_all_mics(room: pra.ShoeBox, table_centers: List[Tuple[float, float]], mics_per_table: int):
    mic_positions_all = []
    for c in table_centers:
        mic_positions_all.append(make_mic_positions(c, mics_per_table, MIC_RADIUS, TABLE_Z))
    mic_positions_all = np.concatenate(mic_positions_all, axis=1)
    room.add_microphone_array(pra.MicrophoneArray(mic_positions_all, SR))


def add_sources_to_room(room: pra.ShoeBox, all_sources: List[Dict[str, Any]]) -> None:
    for s in all_sources:
        room.add_source(s["pos"], signal=s["signal"])


def simulate_sources(room_dims: Tuple[float, float, float], rt60: float, table_centers: List[Tuple[float, float]], mics_per_table: int, all_sources: List[Dict[str, Any]]) -> np.ndarray:
    room = build_room(room_dims, rt60)
    add_all_mics(room, table_centers, mics_per_table)
    add_sources_to_room(room, all_sources)
    room.simulate()
    Y = room.mic_array.signals.astype(np.float32)
    Y = crop_or_pad_2d(Y, T)
    return Y


def simulate_local_targets_fast(
    room: pra.ShoeBox,
    all_sources: List[Dict[str, Any]],
    table_centers: List[Tuple[float, float]],
    mics_per_table: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns per-table metrics at reference mic only:
      target_refclean: (K,T)
      local_energy_frac: (K,)
      bleed_sir_db: (K,)
    Assumes room already has all sources + all mics and room.compute_rir() has run.
    """
    room.compute_rir()
    K = len(table_centers)
    target_refclean = np.zeros((K, T), dtype=np.float32)
    local_energy_frac = np.zeros((K,), dtype=np.float32)
    bleed_sir_db = np.zeros((K,), dtype=np.float32)

    for k in range(K):
        mic_global = k * mics_per_table  # ref mic = first mic of table
        y_local = np.zeros((T,), dtype=np.float32)
        y_other = np.zeros((T,), dtype=np.float32)
        for si, s in enumerate(all_sources):
            h = np.asarray(room.rir[mic_global][si], dtype=np.float32)
            yy = fftconvolve(s["signal"], h, mode="full")[:T].astype(np.float32)
            if s["table"] == k:
                y_local += yy
            else:
                y_other += yy
        target_refclean[k] = y_local
        e_local = energy(y_local)
        e_other = energy(y_other)
        local_energy_frac[k] = float(e_local / (e_local + e_other + 1e-12))
        bleed_sir_db[k] = float(10.0 * np.log10((e_local + 1e-12) / (e_other + 1e-12)))

    return target_refclean, local_energy_frac, bleed_sir_db


# -----------------------------------------------------------------------------
# Corruptions / masking
# -----------------------------------------------------------------------------
def maybe_corrupt_local_mics(
    Y_noisy: np.ndarray,
    rng: random.Random,
    enable: bool,
    prob: float,
    hard_tables: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Corrupt local observed mixture at a subset of tables. Acts only on noisy observation.
    Y_noisy shape: (K,M,T)
    """
    K, M, TT = Y_noisy.shape
    meta: Dict[str, Any] = {"applied": False, "events": []}
    if (not enable) or rng.random() > prob:
        return Y_noisy, meta

    Y = Y_noisy.copy()
    candidate_tables = [int(i) for i in range(K) if hard_tables[i] > 0.5]
    if len(candidate_tables) == 0:
        candidate_tables = list(range(K))
    n_tables = rng.randint(1, max(1, min(2, len(candidate_tables))))
    chosen = rng.sample(candidate_tables, n_tables)

    for k in chosen:
        mode = rng.choice(["gain_drop", "dropout_burst", "noise_burst", "soft_clip"])
        start = rng.randint(0, max(0, TT - SR // 2 - 1))
        dur = rng.randint(SR // 8, SR // 2)
        end = min(TT, start + dur)
        event: Dict[str, Any] = {"table": int(k), "mode": mode, "start": int(start), "end": int(end)}

        for m in range(M):
            if mode == "gain_drop":
                g = rng.uniform(0.08, 0.40)
                Y[k, m, :] *= g
                event["gain"] = float(g)
            elif mode == "dropout_burst":
                g = rng.uniform(0.0, 0.08)
                Y[k, m, start:end] *= g
                event["gain"] = float(g)
            elif mode == "noise_burst":
                seg = Y[k, m, start:end]
                rr = rms(seg)
                noise = rng.normalvariate if False else None
                n = np.random.randn(end - start).astype(np.float32)
                n = n / (rms(n) + 1e-8)
                burst_gain = rng.uniform(0.8, 2.0) * max(rr, 1e-4)
                Y[k, m, start:end] = seg + burst_gain * n
                event["burst_gain"] = float(burst_gain)
            elif mode == "soft_clip":
                drive = rng.uniform(2.0, 5.0)
                Y[k, m, :] = soft_clip(Y[k, m, :], drive=drive)
                event["drive"] = float(drive)
        meta["events"].append(event)

    meta["applied"] = len(meta["events"]) > 0
    return Y.astype(np.float32), meta


def maybe_mask_local_target_observation(
    Y_noisy: np.ndarray,
    target_refclean: np.ndarray,
    rng: random.Random,
    enable: bool,
    prob: float,
    hard_tables: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Make local observation partially miss target information while preserving
    the target label. We approximate this by attenuating segments of the target
    contribution in the observed ref mic only.

    Y_noisy: (K,M,T)
    target_refclean: (K,T)
    """
    K, M, TT = Y_noisy.shape
    meta: Dict[str, Any] = {"applied": False, "events": []}
    if (not enable) or rng.random() > prob:
        return Y_noisy, meta

    Y = Y_noisy.copy()
    candidate_tables = [int(i) for i in range(K) if hard_tables[i] > 0.5]
    if len(candidate_tables) == 0:
        candidate_tables = list(range(K))
    n_tables = rng.randint(1, max(1, min(2, len(candidate_tables))))
    chosen = rng.sample(candidate_tables, n_tables)

    for k in chosen:
        n_bursts = rng.randint(1, 3)
        event = {"table": int(k), "bursts": []}
        for _ in range(n_bursts):
            start = rng.randint(0, max(0, TT - SR // 3 - 1))
            dur = rng.randint(SR // 10, SR // 3)
            end = min(TT, start + dur)
            atten = rng.uniform(0.0, 0.35)
            # only reference mic is modified; neighboring tables remain untouched.
            # remove part of target contribution from observation:
            Y[k, 0, start:end] -= (1.0 - atten) * target_refclean[k, start:end]
            event["bursts"].append({"start": int(start), "end": int(end), "atten": float(atten)})
        meta["events"].append(event)

    meta["applied"] = len(meta["events"]) > 0
    return Y.astype(np.float32), meta


# -----------------------------------------------------------------------------
# Scene generation
# -----------------------------------------------------------------------------
def sample_speakers_per_table(rng: random.Random, min_spk: int, max_spk: int, hard_scene: bool) -> List[int]:
    vals = []
    for _ in range(K_TABLES):
        if hard_scene:
            vals.append(rng.choices(list(range(min_spk, max_spk + 1)), weights=[1, 2, 3][: max_spk - min_spk + 1], k=1)[0])
        else:
            vals.append(rng.randint(min_spk, max_spk))
    return vals


def simulate_one_scene(
    idx: CorpusIndex,
    seed: int,
    mics_per_table: int,
    min_spk_per_table: int,
    max_spk_per_table: int,
    hard_scene_prob: float,
    hard_local_frac_max: float,
    hard_bleed_sir_max_db: float,
    min_hard_tables: int,
    enable_local_mic_corruption: bool,
    mic_corrupt_prob: float,
    enable_local_target_masking: bool,
    target_mask_prob: float,
) -> Optional[Dict[str, Any]]:
    rng = random.Random(seed)
    np.random.seed(seed)

    hard_scene = rng.random() < hard_scene_prob
    layout_name = choose_layout_template(rng, hard_scene_prob=1.0 if hard_scene else 0.0)
    layout = LAYOUT_TEMPLATES[layout_name]
    room_dims = tuple(layout["room_dims"])
    table_centers = jitter_layout(layout["table_centers"], room_dims, rng, jitter_xy=0.22 if hard_scene else 0.10)
    adj = make_adjacency(table_centers, thresh_m=float(layout["adj_thresh_m"]))

    rt60 = rng.uniform(*RT60_RANGE)
    snr_db = rng.uniform(*SNR_DB_RANGE)
    speakers_per_table = sample_speakers_per_table(rng, min_spk_per_table, max_spk_per_table, hard_scene)

    # Build all sources.
    all_sources: List[Dict[str, Any]] = []
    for k in range(K_TABLES):
        for p in range(speakers_per_table[k]):
            sp_path = rng.choice(idx.speech_paths)
            sig = load_audio_16k(sp_path, T, rng)

            # Slightly larger amplitude variance in hard scenes.
            sig *= float(rng.uniform(0.7, 1.4) if hard_scene else rng.uniform(0.85, 1.15))
            sig = normalize_rms(sig, target_rms=float(rng.uniform(0.035, 0.075) if hard_scene else rng.uniform(0.04, 0.065)))

            pos = sample_speaker_position(table_centers[k], SPK_RADIUS_RANGE, TABLE_Z, rng)
            all_sources.append({
                "table": int(k),
                "spk": int(p),
                "path": sp_path,
                "pos": [float(v) for v in pos],
                "signal": sig.astype(np.float32),
            })

    # Simulate clean reverberant mixture at all mics.
    room = build_room(room_dims, rt60)
    add_all_mics(room, table_centers, mics_per_table)
    add_sources_to_room(room, all_sources)
    room.simulate()
    Y_clean = crop_or_pad_2d(room.mic_array.signals.astype(np.float32), T)
    Y_clean = Y_clean.reshape(K_TABLES, mics_per_table, T)

    target_refclean, local_energy_frac, bleed_sir_db = simulate_local_targets_fast(
        room=room,
        all_sources=all_sources,
        table_centers=table_centers,
        mics_per_table=mics_per_table,
    )

    is_hard_table = np.logical_or(local_energy_frac <= hard_local_frac_max, bleed_sir_db <= hard_bleed_sir_max_db).astype(np.float32)
    n_hard_tables = int(is_hard_table.sum())
    if hard_scene and n_hard_tables < int(min_hard_tables):
        return None

    # Add external noise.
    noise_path = choose_noise_track(idx, rng)
    noise_mono = load_audio_16k(noise_path, T, rng)
    # shared ambient + small per-mic perturbation
    noise = np.stack([noise_mono + 0.03 * np.random.randn(T).astype(np.float32) for _ in range(K_TABLES * mics_per_table)], axis=0)
    noise = noise.reshape(K_TABLES, mics_per_table, T)
    g_noise = scale_to_target_snr(Y_clean[:, 0, :].reshape(-1), noise[:, 0, :].reshape(-1), snr_db)
    Y_noisy = Y_clean + g_noise * noise

    # Make local observation harder only at observed mixture.
    local_corruption_meta = {"applied": False, "events": []}
    target_mask_meta = {"applied": False, "events": []}
    Y_noisy, local_corruption_meta = maybe_corrupt_local_mics(
        Y_noisy=Y_noisy,
        rng=rng,
        enable=enable_local_mic_corruption,
        prob=mic_corrupt_prob,
        hard_tables=is_hard_table,
    )
    Y_noisy, target_mask_meta = maybe_mask_local_target_observation(
        Y_noisy=Y_noisy,
        target_refclean=target_refclean,
        rng=rng,
        enable=enable_local_target_masking,
        prob=target_mask_prob,
        hard_tables=is_hard_table,
    )

    table_meta = []
    corrupted_tables = {int(ev["table"]) for ev in local_corruption_meta.get("events", [])}
    masked_tables = {int(ev["table"]) for ev in target_mask_meta.get("events", [])}
    for k in range(K_TABLES):
        table_meta.append({
            "table": int(k),
            "n_local_speakers": int(speakers_per_table[k]),
            "local_energy_frac": float(local_energy_frac[k]),
            "bleed_sir_db": float(bleed_sir_db[k]),
            "is_hard_table": bool(is_hard_table[k] > 0.5),
            "local_mic_corrupted": bool(k in corrupted_tables),
            "local_target_masked": bool(k in masked_tables),
        })

    # drop source waveforms from metadata copy
    src_meta = []
    for s in all_sources:
        src_meta.append({k: v for k, v in s.items() if k != "signal"})

    return {
        "Y": Y_noisy.astype(np.float32),
        "target_refclean": target_refclean.astype(np.float32),
        "adj": adj.astype(np.float32),
        "rt60": float(rt60),
        "snr_db": float(snr_db),
        "seed": int(seed),
        "meta": {
            "seed": int(seed),
            "snr_db": float(snr_db),
            "rt60": float(rt60),
            "layout_name": layout_name,
            "room_dims": [float(v) for v in room_dims],
            "table_centers": [[float(a), float(b)] for a, b in table_centers],
            "adj_thresh_m": float(layout["adj_thresh_m"]),
            "speakers_per_table": [int(v) for v in speakers_per_table],
            "local_energy_frac": [float(v) for v in local_energy_frac.tolist()],
            "bleed_sir_db": [float(v) for v in bleed_sir_db.tolist()],
            "is_hard_table": [bool(v > 0.5) for v in is_hard_table.tolist()],
            "n_hard_tables": int(n_hard_tables),
            "hard_scene_requested": bool(hard_scene),
            "hard_scene_accepted": bool(n_hard_tables >= int(min_hard_tables)) if hard_scene else False,
            "noise_path": noise_path,
            "local_mic_corruption": local_corruption_meta,
            "local_target_masking": target_mask_meta,
            "tables": table_meta,
            "sources": src_meta,
        },
    }


# -----------------------------------------------------------------------------
# Split generation / writing
# -----------------------------------------------------------------------------
def write_shard(out_prefix: str, scenes: List[Dict[str, Any]]) -> None:
    B = len(scenes)
    Y = np.stack([s["Y"] for s in scenes], axis=0)
    tgt = np.stack([s["target_refclean"] for s in scenes], axis=0)
    adj = np.stack([s["adj"] for s in scenes], axis=0)
    rt60 = np.asarray([s["rt60"] for s in scenes], dtype=np.float32)
    snr_db = np.asarray([s["snr_db"] for s in scenes], dtype=np.float32)
    seed = np.asarray([s["seed"] for s in scenes], dtype=np.int64)

    np.savez_compressed(
        out_prefix + ".npz",
        Y=Y,
        target_refclean=tgt,
        adj=adj,
        rt60=rt60,
        snr_db=snr_db,
        seed=seed,
    )

    with open(out_prefix + ".jsonl", "w") as f:
        for s in scenes:
            f.write(json.dumps(s["meta"]) + "\n")


def build_split(
    split_name: str,
    out_dir: str,
    n_examples: int,
    shard_size: int,
    base_seed: int,
    **scene_kwargs,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    scenes_buf: List[Dict[str, Any]] = []
    shard_id = 0
    produced = 0
    tries = 0
    pbar = tqdm(total=n_examples, desc=f"Building {split_name}", ncols=100)

    while produced < n_examples:
        seed = base_seed + tries
        tries += 1
        scene = simulate_one_scene(seed=seed, **scene_kwargs)
        if scene is None:
            continue
        scenes_buf.append(scene)
        produced += 1
        pbar.update(1)

        if len(scenes_buf) >= shard_size or produced == n_examples:
            prefix = os.path.join(out_dir, f"shard_{shard_id:04d}")
            write_shard(prefix, scenes_buf)
            shard_id += 1
            scenes_buf = []
    pbar.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    global K_TABLES, LAYOUT_TEMPLATES
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--shard_size", type=int, default=50)

    ap.add_argument("--libri_root", type=str, default=LIBRI_DEFAULT)
    ap.add_argument("--vctk_root", type=str, default=VCTK_DEFAULT)
    ap.add_argument("--musan_root", type=str, default=MUSAN_DEFAULT)

    ap.add_argument("--mics_per_table", type=int, default=DEFAULT_MICS_PER_TABLE)
    ap.add_argument("--n_tables", type=int, default=K_TABLES, choices=[4, 6],
                    help="Number of tables/nodes. K=4 reproduces the original setup; K=6 uses six-table layouts.")
    ap.add_argument("--min_speakers_per_table", type=int, default=1)
    ap.add_argument("--max_speakers_per_table", type=int, default=3)

    ap.add_argument("--hard_scene_prob", type=float, default=0.90)
    ap.add_argument("--hard_local_frac_max", type=float, default=DEFAULT_HARD_LOCAL_FRAC_MAX)
    ap.add_argument("--hard_bleed_sir_max_db", type=float, default=DEFAULT_HARD_BLEED_SIR_MAX_DB)
    ap.add_argument("--min_hard_tables", type=int, default=DEFAULT_MIN_HARD_TABLES)

    ap.add_argument("--enable_local_mic_corruption", action="store_true")
    ap.add_argument("--mic_corrupt_prob", type=float, default=0.35)
    ap.add_argument("--enable_local_target_masking", action="store_true")
    ap.add_argument("--target_mask_prob", type=float, default=0.25)

    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    K_TABLES = int(args.n_tables)
    if K_TABLES == 6:
        LAYOUT_TEMPLATES = LAYOUT_TEMPLATES_K6

    if args.min_speakers_per_table < 1 or args.max_speakers_per_table < args.min_speakers_per_table:
        raise ValueError("Invalid speaker-per-table range")
    if args.mics_per_table < 1:
        raise ValueError("mics_per_table must be >= 1")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "test"), exist_ok=True)

    idx = build_index(args.libri_root, args.vctk_root, args.musan_root)

    manifest = {
        "name": "RestaurantSim_v6_graph_required",
        "description": "Harder restaurant ASN dataset with coupled layouts, hard-scene filtering, local mic corruption, and local target masking to make cross-node information more useful.",
        "sr": SR,
        "duration_s": DURATION_S,
        "T": T,
        "K": K_TABLES,
        "M": int(args.mics_per_table),
        "min_speakers_per_table": int(args.min_speakers_per_table),
        "max_speakers_per_table": int(args.max_speakers_per_table),
        "rt60_range": list(RT60_RANGE),
        "snr_db_range": list(SNR_DB_RANGE),
        "hard_scene_prob": float(args.hard_scene_prob),
        "hard_local_frac_max": float(args.hard_local_frac_max),
        "hard_bleed_sir_max_db": float(args.hard_bleed_sir_max_db),
        "min_hard_tables": int(args.min_hard_tables),
        "enable_local_mic_corruption": bool(args.enable_local_mic_corruption),
        "mic_corrupt_prob": float(args.mic_corrupt_prob),
        "enable_local_target_masking": bool(args.enable_local_target_masking),
        "target_mask_prob": float(args.target_mask_prob),
        "layout_templates": LAYOUT_TEMPLATES,
        "libri_root": args.libri_root,
        "vctk_root": args.vctk_root,
        "musan_root": args.musan_root,
        "seed": int(args.seed),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    common_kwargs = dict(
        idx=idx,
        mics_per_table=int(args.mics_per_table),
        min_spk_per_table=int(args.min_speakers_per_table),
        max_spk_per_table=int(args.max_speakers_per_table),
        hard_scene_prob=float(args.hard_scene_prob),
        hard_local_frac_max=float(args.hard_local_frac_max),
        hard_bleed_sir_max_db=float(args.hard_bleed_sir_max_db),
        min_hard_tables=int(args.min_hard_tables),
        enable_local_mic_corruption=bool(args.enable_local_mic_corruption),
        mic_corrupt_prob=float(args.mic_corrupt_prob),
        enable_local_target_masking=bool(args.enable_local_target_masking),
        target_mask_prob=float(args.target_mask_prob),
    )

    build_split(
        split_name="train",
        out_dir=os.path.join(args.out, "train"),
        n_examples=int(args.n_train),
        shard_size=int(args.shard_size),
        base_seed=int(args.seed) + 0,
        **common_kwargs,
    )
    build_split(
        split_name="val",
        out_dir=os.path.join(args.out, "val"),
        n_examples=int(args.n_val),
        shard_size=int(args.shard_size),
        base_seed=int(args.seed) + 1000000,
        **common_kwargs,
    )
    build_split(
        split_name="test",
        out_dir=os.path.join(args.out, "test"),
        n_examples=int(args.n_test),
        shard_size=int(args.shard_size),
        base_seed=int(args.seed) + 2000000,
        **common_kwargs,
    )

    print("Done.")
    print(f"Manifest saved to: {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
