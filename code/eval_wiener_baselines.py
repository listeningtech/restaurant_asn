#!/usr/bin/env python3
"""
eval_wiener_baselines.py

Evaluates classical Wiener-filter baselines on the restaurant-scene test set.

Algorithms
──────────
  unprocessed   Raw mic-0 signal at each node (no processing)
  local         Per-node single-channel Wiener filter (no cross-node comms)
  centralized   Oracle K-mic centralized MWF — the upper bound
  danse         Vanilla DANSE, sequential node updating [Bertrand & Moonen, TSP 2010]
  dmwf          Distributed MWF, non-iterative [Didier et al., arXiv 2603.09735, 2026]

All methods estimate the Wiener filter from oracle cross-covariance matrices
computed using the available target_refclean signals. This gives each classical
method the best possible SCM estimate — a fair upper-bound comparison against
the trained neural model.

Assumptions (matching the dataset)
───────────────────────────────────
  K  = 4 nodes,  M = 1 mic per node
  T  = 80 000 samples (5 s @ 16 kHz)
  STFT: n_fft = 512, hop = 128, Hann window  (matches training script)
  Fully-connected WASN: all K nodes communicate (DANSE / dMWF papers assumption)

Usage
─────
  /home/rrame12/anaconda3/envs/torch/bin/python3 eval_wiener_baselines.py \\
      --data /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk \\
      --split test --nfft 512 --hop 128 --danse_iters 30
"""

import os
import glob
import json
import argparse

import numpy as np
import scipy.signal as ss
from tqdm import tqdm

SR  = 16_000
EPS = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# STFT / iSTFT  (scipy, Hann window, matches torch STFT centre=True spirit)
# ─────────────────────────────────────────────────────────────────────────────

def _stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """x: (T,) float64  →  complex (F, TT)   F = n_fft//2 + 1"""
    win = np.hanning(n_fft)
    _, _, Z = ss.stft(x, nperseg=n_fft, noverlap=n_fft - hop,
                      window=win, padded=True, boundary="zeros")
    return Z  # (F, TT)


def _istft(X: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    """X: (F, TT) complex  →  (length,) float64"""
    win = np.hanning(n_fft)
    _, x = ss.istft(X, nperseg=n_fft, noverlap=n_fft - hop,
                    window=win, boundary=True)
    return x[:length]


def stft_multichannel(Y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Y: (K, T) float64  →  complex (K, F, TT)"""
    return np.stack([_stft(Y[k], n_fft, hop) for k in range(Y.shape[0])])


# ─────────────────────────────────────────────────────────────────────────────
# Covariance estimation  (oracle: uses clean targets directly)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_scms(Y_F: np.ndarray, D_F: np.ndarray):
    """
    Compute spatial covariance matrices from the full clip.

    Parameters
    ----------
    Y_F : (K, F, TT) complex — noisy observations
    D_F : (K, F, TT) complex — oracle clean targets

    Returns
    -------
    Ryy : (F, K, K) complex  —  E[y y^H]  per freq bin
    Ryd : (F, K, K) complex  —  Ryd[:, :, k] = E[y d_k^H]  per freq bin
    """
    TT = Y_F.shape[2]
    Yt = Y_F.transpose(1, 0, 2)   # (F, K, TT)
    Dt = D_F.transpose(1, 0, 2)   # (F, K, TT)
    Ryy = (Yt @ Yt.conj().transpose(0, 2, 1)) / TT   # (F, K, K)
    Ryd = (Yt @ Dt.conj().transpose(0, 2, 1)) / TT   # (F, K, K)
    return Ryy, Ryd


def regularize(Ryy: np.ndarray, beta: float = 1e-3) -> np.ndarray:
    """Diagonal loading: Ryy + beta * mean_diag * I  (per freq bin)."""
    K  = Ryy.shape[-1]
    tr = np.real(np.trace(Ryy, axis1=-2, axis2=-1)).mean() / K + EPS
    return Ryy + beta * tr * np.eye(K, dtype=complex)[np.newaxis]   # (F,K,K)


# ─────────────────────────────────────────────────────────────────────────────
# Filter application
# ─────────────────────────────────────────────────────────────────────────────

def apply_filter(W: np.ndarray, Y_aug: np.ndarray) -> np.ndarray:
    """
    d̂[f,t] = W[f]^H @ Y_aug[f,:,t]  =  Σ_k  conj(W[f,k]) * Y_aug[f,k,t]

    Parameters
    ----------
    W     : (F, K_in) complex
    Y_aug : (F, K_in, TT) complex

    Returns
    -------
    (F, TT) complex estimated STFT
    """
    return np.einsum("fk,fkt->ft", W.conj(), Y_aug)


# ─────────────────────────────────────────────────────────────────────────────
# SI-SDR metric
# ─────────────────────────────────────────────────────────────────────────────

def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB.  est, ref: (T,) float"""
    est = est - est.mean()
    ref = ref - ref.mean()
    dot     = float(np.dot(est, ref))
    ref_e   = float(np.dot(ref, ref)) + eps
    s_tgt   = (dot / ref_e) * ref
    e_noise = est - s_tgt
    return 10.0 * np.log10(
        (float(np.dot(s_tgt, s_tgt)) + eps) /
        (float(np.dot(e_noise, e_noise)) + eps)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1 — Local single-channel MWF (no cross-node)
# ─────────────────────────────────────────────────────────────────────────────

def run_local(Y_F, Ryy, Ryd, n_fft, hop, T, K):
    """
    At each node k independently:
        W_k[f] = R_{y_k d_k}[f] / R_{y_k y_k}[f]    (scalar Wiener filter)
        d̂_k[f,t] = conj(W_k[f]) * y_k[f,t]
    """
    Yt   = Y_F.transpose(1, 0, 2)   # (F, K, TT)
    ests = []
    for k in range(K):
        W_scalar = Ryd[:, k, k] / (Ryy[:, k, k] + EPS)   # (F,)
        D_hat    = W_scalar.conj()[:, None] * Yt[:, k, :]  # (F, TT)
        ests.append(_istft(D_hat, n_fft, hop, T))
    return np.stack(ests)   # (K, T)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 2 — Centralized MWF (oracle upper bound)
# ─────────────────────────────────────────────────────────────────────────────

def run_centralized(Y_F, Ryy, Ryd, n_fft, hop, T, K):
    """
    Uses all K nodes' signals jointly:
        W_k[f] = R_yy[f]^{-1} R_{yd_k}[f]   ∈ C^K
        d̂_k[f,t] = W_k[f]^H y[f,t]
    """
    Ryy_r = regularize(Ryy)              # (F, K, K)
    Yt    = Y_F.transpose(1, 0, 2)       # (F, K, TT)
    ests  = []
    for k in range(K):
        W = np.linalg.solve(Ryy_r, Ryd[:, :, k:k+1]).squeeze(-1)   # (F, K)
        ests.append(_istft(apply_filter(W, Yt), n_fft, hop, T))
    return np.stack(ests)   # (K, T)


# ─────────────────────────────────────────────────────────────────────────────
# DANSE / dMWF shared helper — augmented observation transform
# ─────────────────────────────────────────────────────────────────────────────

def _build_Tk(P_conj: np.ndarray, k: int, K: int) -> np.ndarray:
    """
    Build T_k ∈ C^{F×K×K} that maps centralized y → augmented ỹ_k.

    With M=1 per node and Q=1 fused channel, the augmented observation is:
        ỹ_k = [y_k ; P_1* y_1 ; … ; P_{k-1}* y_{k-1} ;
                     P_{k+1}* y_{k+1} ; … ; P_K* y_K]

    Row 0 of T_k selects y_k (local signal, always first).
    Row j (j≥1) selects P_{q_j}* y_{q_j} for the j-th neighbour q_j ≠ k.

    Parameters
    ----------
    P_conj : (F, K) complex — element-wise conjugates of fusion scalars
    k      : updating / estimating node index
    K      : total number of nodes

    Returns
    -------
    T_k : (F, K, K) complex
    """
    F  = P_conj.shape[0]
    Tk = np.zeros((F, K, K), dtype=complex)
    Tk[:, 0, k] = 1.0   # row 0 → y_k
    row = 1
    for q in range(K):
        if q != k:
            Tk[:, row, q] = P_conj[:, q]   # row `row` → P_q* y_q
            row += 1
    return Tk   # (F, K, K)


def _solve_aug_filter(Ryy_r: np.ndarray, Ryd_k: np.ndarray,
                      Tk: np.ndarray, K: int) -> np.ndarray:
    """
    Compute the augmented Wiener filter W̃_k for one node k.

        R_{ỹ_k ỹ_k} = T_k  R_yy  T_k^H
        R_{ỹ_k d_k} = T_k  R_{yd_k}
        W̃_k          = R_{ỹ_k ỹ_k}^{-1}  R_{ỹ_k d_k}

    Parameters
    ----------
    Ryy_r : (F, K, K) complex — regularized centralized SCM
    Ryd_k : (F, K, 1) complex — centralized cross-SCM for node k
    Tk    : (F, K, K) complex — transform matrix for node k

    Returns
    -------
    W̃_k : (F, K) complex
    """
    TkH       = Tk.conj().transpose(0, 2, 1)          # (F, K, K)
    R_tilde   = Tk @ Ryy_r @ TkH                      # (F, K, K)
    Rtilde_dk = Tk @ Ryd_k                            # (F, K, 1)
    R_tilde  += EPS * np.eye(K, dtype=complex)[None]  # extra guard
    return np.linalg.solve(R_tilde, Rtilde_dk).squeeze(-1)   # (F, K)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 3 — Vanilla DANSE (sequential node updating, Q=1)
# ─────────────────────────────────────────────────────────────────────────────

def run_danse(Y_F, Ryy, Ryd, n_fft, hop, T, K, n_iters: int = 30):
    """
    Distributed Adaptive Node-Specific Signal Estimation.
    Reference: Bertrand & Moonen, IEEE TSP 58(10), 2010, Part I.

    With M=1 per node and Q=1 fused channel:
      • Each node k sends z_k = P_k* y_k  (P_k is a complex scalar per freq bin)
      • Node k observes ỹ_k = [y_k ; {P_q* y_q}_{q≠k}]
      • Sequential update: for each k, solve LMMSE on ỹ_k; set P_k = W̃_k[0]

    Converges to centralized MWF in FODS scenarios.
    In this PODS scenario (local-only desired signals) DANSE is sub-optimal.
    """
    F     = Y_F.shape[1]
    Ryy_r = regularize(Ryy)          # (F, K, K)
    Yt    = Y_F.transpose(1, 0, 2)   # (F, K, TT)

    # Initialise fusion scalars to 1 (identity / unweighted broadcast)
    P = np.ones((F, K), dtype=complex)

    # Store the final per-node filter (updated each sub-iteration)
    W_final = np.zeros((K, F, K), dtype=complex)

    for _ in range(n_iters):
        for k in range(K):
            Tk           = _build_Tk(P.conj(), k, K)                  # (F,K,K)
            W_tilde      = _solve_aug_filter(Ryy_r,
                                             Ryd[:, :, k:k+1], Tk, K)  # (F,K)
            P[:, k]      = W_tilde[:, 0]   # local part → new fusion scalar
            W_final[k]   = W_tilde

    # Apply final filters
    ests = []
    for k in range(K):
        Tk    = _build_Tk(P.conj(), k, K)
        Yt_k  = np.einsum("fij,fjt->fit", Tk, Yt)    # (F, K, TT)
        D_hat = apply_filter(W_final[k], Yt_k)        # (F, TT)
        ests.append(_istft(D_hat, n_fft, hop, T))
    return np.stack(ests)   # (K, T)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 4 — dMWF (non-iterative, one-shot)
# ─────────────────────────────────────────────────────────────────────────────

def run_dmwf(Y_F, Ryy, Ryd, n_fft, hop, T, K):
    """
    Distributed Multichannel Wiener Filter.
    Reference: Didier et al., arXiv 2603.09735, IEEE 2026.

    With M=1 per node (Q̊_q = 1 for all q):

    ── Discovery step (Eq. 29) ──────────────────────────────────────────────
        ρ_q = Σ_{k≠q} y_k               (sum of all neighbours' signals)
        R_{y_q, ρ_q}[f] = Σ_{k≠q} R_yy[f, q, k]
        P_q[f] = R_{y_q, ρ_q}[f] / R_{y_q, y_q}[f]

    ── Estimation step (Eq. 18–19) ──────────────────────────────────────────
        z_q = P_q* y_q   (fused signal sent by node q)
        ỹ_k = [y_k ; {z_q}_{q≠k}]
        W̃_k = R_{ỹ_k ỹ_k}^{-1} R_{ỹ_k d_k}

    One-shot optimal in the PODS scenario (Theorem 1 of the paper).
    """
    F     = Y_F.shape[1]
    Ryy_r = regularize(Ryy)          # (F, K, K)
    Yt    = Y_F.transpose(1, 0, 2)   # (F, K, TT)

    # ── Discovery: closed-form fusion scalars ────────────────────────────────
    # R_{y_q, ρ_q}[f] = (row q of Ryy[f]) summed over k≠q
    #                  = Ryy[f, q, :].sum() − Ryy[f, q, q]
    row_sums = Ryy.sum(axis=-1)                           # (F, K)
    diag     = Ryy[:, np.arange(K), np.arange(K)]        # (F, K)
    R_cross  = row_sums - diag                            # (F, K)  = Σ_{k≠q} Ryy[f,q,k]
    P        = R_cross / (diag + EPS)                     # (F, K)  = P_q[f] per eq.(29)

    # ── Estimation: one MWF solve per node ──────────────────────────────────
    ests = []
    for k in range(K):
        Tk    = _build_Tk(P.conj(), k, K)                         # (F, K, K)
        W     = _solve_aug_filter(Ryy_r, Ryd[:, :, k:k+1], Tk, K)  # (F, K)
        Yt_k  = np.einsum("fij,fjt->fit", Tk, Yt)                 # (F, K, TT)
        D_hat = apply_filter(W, Yt_k)                             # (F, TT)
        ests.append(_istft(D_hat, n_fft, hop, T))
    return np.stack(ests)   # (K, T)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

ALGOS = {
    "local":       run_local,
    "centralized": run_centralized,
    "danse":       run_danse,
    "dmwf":        run_dmwf,
}


def eval_shard(path, algos, n_fft, hop, K, n_iters, results):
    data  = np.load(path)
    Y_all = data["Y"]                # (B, K, M, T) float32
    D_all = data["target_refclean"] # (B, K, T)    float32
    B     = Y_all.shape[0]

    for b in range(B):
        Y = Y_all[b, :, 0, :].astype(np.float64)   # (K, T) — mic-0 only (M=1)
        D = D_all[b].astype(np.float64)              # (K, T)
        T = Y.shape[1]

        # Unprocessed SI-SDR (mic-0 vs clean target)
        results["unprocessed"].append(
            float(np.mean([si_sdr(Y[k], D[k]) for k in range(K)]))
        )

        # STFT of observations and oracle targets
        Y_F = stft_multichannel(Y, n_fft, hop)   # (K, F, TT)
        D_F = stft_multichannel(D, n_fft, hop)   # (K, F, TT)

        # Oracle SCMs (full-clip batch estimate)
        Ryy, Ryd = estimate_scms(Y_F, D_F)

        for name, fn in algos.items():
            if name == "danse":
                ests = fn(Y_F, Ryy, Ryd, n_fft, hop, T, K, n_iters)
            else:
                ests = fn(Y_F, Ryy, Ryd, n_fft, hop, T, K)

            results[name].append(
                float(np.mean([si_sdr(ests[k], D[k]) for k in range(K)]))
            )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Classical Wiener-filter baselines")
    ap.add_argument("--data",        required=True,
                    help="Dataset root (must contain <split>/shard_*.npz)")
    ap.add_argument("--split",       default="test",
                    help="Which split to evaluate (default: test)")
    ap.add_argument("--nfft",        type=int,   default=512)
    ap.add_argument("--hop",         type=int,   default=128)
    ap.add_argument("--K",           type=int,   default=4,
                    help="Number of nodes (default: 4)")
    ap.add_argument("--danse_iters", type=int,   default=30,
                    help="DANSE sequential update iterations (default: 30)")
    ap.add_argument("--algos",       nargs="+",
                    default=["local", "centralized", "danse", "dmwf"],
                    choices=list(ALGOS.keys()))
    ap.add_argument("--out",         default=None,
                    help="Optional JSON path to save results")
    args = ap.parse_args()

    split_dir   = os.path.join(args.data, args.split)
    shard_paths = sorted(glob.glob(os.path.join(split_dir, "shard_*.npz")))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {split_dir}")
    print(f"Evaluating {len(shard_paths)} shards  ({args.split} split)  "
          f"K={args.K}  n_fft={args.nfft}  hop={args.hop}  "
          f"danse_iters={args.danse_iters}")

    algos   = {a: ALGOS[a] for a in args.algos}
    results = {"unprocessed": [], **{a: [] for a in args.algos}}

    for sp in tqdm(shard_paths, desc="Shards", ncols=80):
        eval_shard(sp, algos, args.nfft, args.hop, args.K,
                   args.danse_iters, results)

    # ── Report ──────────────────────────────────────────────────────────────
    n       = len(results["unprocessed"])
    in_mean = float(np.mean(results["unprocessed"]))
    in_std  = float(np.std(results["unprocessed"]))

    print(f"\nResults over {n} examples")
    print(f"{'Algorithm':<16}  {'SI-SDR (dB)':>13}  {'ΔSI-SDR':>10}")
    print("─" * 44)
    print(f"{'unprocessed':<16}  {in_mean:>+7.2f} ±{in_std:>4.2f}  {'—':>10}")
    for name in args.algos:
        vals    = np.array(results[name])
        out_m   = float(vals.mean())
        out_std = float(vals.std())
        delta   = out_m - in_mean
        print(f"{name:<16}  {out_m:>+7.2f} ±{out_std:>4.2f}  {delta:>+10.2f}")

    if args.out:
        summary = {
            "split":       args.split,
            "n_examples":  n,
            "nfft":        args.nfft,
            "hop":         args.hop,
            "K":           args.K,
            "danse_iters": args.danse_iters,
            "algorithms":  args.algos,
            "results": {
                "unprocessed": {
                    "mean": in_mean, "std": in_std,
                    "values": results["unprocessed"],
                },
                **{
                    a: {
                        "mean":   float(np.mean(results[a])),
                        "std":    float(np.std(results[a])),
                        "delta":  float(np.mean(results[a])) - in_mean,
                        "values": results[a],
                    }
                    for a in args.algos
                },
            },
        }
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
