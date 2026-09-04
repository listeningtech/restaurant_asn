#!/usr/bin/env python3
"""
train_graph_ablations_v8.py

Two ablations extending V8 (train_graph_input_crossnode_v8.py):

  (iii) --adj_mode alltoall
        Replace the stored per-scene adjacency with a fully-connected graph:
        every node sees every other node's log-magnitude.
        Tests whether the designated pairing (0↔2, 1↔3) actually matters,
        or whether just having any neighbour's spectrum is sufficient.
        (The dataset has only one fixed adj pattern, so this is the meaningful
        topology ablation.)

  (iv)  --inject_level1
        Also inject neighbour log-magnitude at encoder level 1 (resolution
        ~129×126, i.e. after one stride-2 downsample from 257×251).
        A lightweight 1×1 conv projects (M → enc_chs[0]) and adds residually,
        zero-initialised so training starts from the V8 baseline.
        Tests whether reinforcing the neighbour signal after the first
        nonlinearity provides additional benefit.

All other flags are identical to V8.  The graph-OFF baseline is
train_graph_input_crossnode_v8.py with --disable_graph.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ablation run commands
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(iii) all-to-all adjacency:
  CUDA_VISIBLE_DEVICES=1 python train_graph_ablations_v8.py \\
    --data /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk \\
    --out  /home/rrame12/Desktop/Research/ASN/runs_ablation_v8_alltoall \\
    --batch 8 --epochs 120 --lr 3e-4 --patience 20 \\
    --base 32 --depth 4 --crop_s 2.0 --nfft 512 --hop 128 \\
    --adj_mode alltoall

(iv) multi-level injection:
  CUDA_VISIBLE_DEVICES=2 python train_graph_ablations_v8.py \\
    --data /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk \\
    --out  /home/rrame12/Desktop/Research/ASN/runs_ablation_v8_multilevel \\
    --batch 8 --epochs 120 --lr 3e-4 --patience 20 \\
    --base 32 --depth 4 --crop_s 2.0 --nfft 512 --hop 128 \\
    --inject_level1

(iii)+(iv) combined:
  CUDA_VISIBLE_DEVICES=3 python train_graph_ablations_v8.py \\
    --data /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk \\
    --out  /home/rrame12/Desktop/Research/ASN/runs_ablation_v8_combined \\
    --batch 8 --epochs 120 --lr 3e-4 --patience 20 \\
    --base 32 --depth 4 --crop_s 2.0 --nfft 512 --hop 128 \\
    --adj_mode alltoall --inject_level1
"""

import os
import glob
import json
import random
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SR  = 16000
EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Utilities  (identical to V8)
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def compute_si_sdr(est, ref, eps=EPS):
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    ref_energy = (ref * ref).sum(dim=-1, keepdim=True) + eps
    s_target   = ((est * ref).sum(dim=-1, keepdim=True) / ref_energy) * ref
    e_noise    = est - s_target
    return 10.0 * torch.log10((s_target*s_target).sum(dim=-1) /
                               ((e_noise*e_noise).sum(dim=-1) + eps) + eps)

def si_sdr_loss(est, ref, eps=EPS):
    return -compute_si_sdr(est, ref, eps=eps).mean()

@torch.no_grad()
def compute_si_sdr_metric(est, ref, eps=EPS):
    return compute_si_sdr(est, ref, eps=eps)

def read_json(path):
    with open(path) as f: return json.load(f)

def write_json(path, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def resolve_n_mics(data_root, override=None):
    if override is not None: return int(override)
    m = read_json(os.path.join(data_root, "manifest.json"))
    return int(m["M"])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset  (identical to V8)
# ─────────────────────────────────────────────────────────────────────────────

class RestaurantSceneDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir, crop_samples, cache=False):
        super().__init__()
        self.crop_samples = crop_samples
        self.cache        = cache
        self.shard_paths  = sorted(glob.glob(os.path.join(split_dir, "shard_*.npz")))
        if not self.shard_paths:
            raise RuntimeError(f"No shards found in {split_dir}")
        self.index = []
        for sid, sp in enumerate(self.shard_paths):
            with np.load(sp) as d:
                for i in range(d["Y"].shape[0]):
                    self.index.append((sid, i))
        self._ram_cache = {}

    def __len__(self): return len(self.index)

    def _load(self, sid):
        if self.cache and sid in self._ram_cache:
            return self._ram_cache[sid]
        d = np.load(self.shard_paths[sid])
        sh = {k: d[k] for k in d.files}; d.close()
        if self.cache: self._ram_cache[sid] = sh
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
        return {"Y": torch.from_numpy(Y).float(),
                "S": torch.from_numpy(S).float(),
                "A": torch.from_numpy(A).float()}

def collate_fn(batch):
    return {"Y": torch.stack([b["Y"] for b in batch], 0),
            "S": torch.stack([b["S"] for b in batch], 0),
            "A": torch.stack([b["A"] for b in batch], 0)}


# ─────────────────────────────────────────────────────────────────────────────
# STFT
# ─────────────────────────────────────────────────────────────────────────────

class TorchSTFT(nn.Module):
    def __init__(self, n_fft=512, hop=128, win_length=None):
        super().__init__()
        self.n_fft = n_fft; self.hop = hop
        wl = win_length if win_length is not None else n_fft
        self.win_length = wl
        self.register_buffer("window", torch.hann_window(wl))

    def stft(self, x):
        N, M, T = x.shape
        X = torch.stft(x.reshape(N*M, T), self.n_fft, self.hop, self.win_length,
                       window=self.window, center=True, return_complex=True)
        Freq, TT = X.shape[-2], X.shape[-1]
        return X.reshape(N, M, Freq, TT)

    def istft(self, X, length):
        return torch.istft(X, self.n_fft, self.hop, self.win_length,
                           window=self.window, center=True, length=length)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvGNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, drop=0.0):
        super().__init__()
        ng = min(8, cout)
        while cout % ng != 0 and ng > 1: ng -= 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p), nn.GroupNorm(ng, cout), nn.PReLU(),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity())
    def forward(self, x): return self.net(x)

class DownBlock(nn.Module):
    def __init__(self, cin, cout, drop=0.0):
        super().__init__()
        self.c1   = ConvGNAct(cin, cout, drop=drop)
        self.c2   = ConvGNAct(cout, cout, drop=drop)
        self.down = nn.Conv2d(cout, cout, 3, stride=2, padding=1)
    def forward(self, x):
        x = self.c2(self.c1(x)); skip = x; x = self.down(x)
        return skip, x

class UpBlock(nn.Module):
    def __init__(self, cin, skip_ch, cout, drop=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.c1 = ConvGNAct(cout + skip_ch, cout, drop=drop)
        self.c2 = ConvGNAct(cout, cout, drop=drop)
    def forward(self, x, skip):
        x  = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]; dw = skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, (0, max(dw,0), 0, max(dh,0)))
            x = x[..., :skip.shape[-2], :skip.shape[-1]]
        return self.c2(self.c1(torch.cat([x, skip], dim=1)))


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck FiLM  (carried over from V8, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class CrossNodeSpectralAttn(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads; self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5; self.dropout = dropout
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, A):
        B, K, C, Freq = x.shape; H, D = self.heads, self.head_dim
        eye  = torch.eye(K, device=A.device, dtype=A.dtype).unsqueeze(0)
        A    = (A + eye).clamp(0.0, 1.0); mask = (A > 0.0)
        x_bf = x.permute(0,3,1,2).reshape(B*Freq, K, C)
        x_bf = self.norm(x_bf)
        qkv  = self.qkv(x_bf).reshape(B*Freq, K, 3, H, D)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1,2); k = k.transpose(1,2); v = v.transpose(1,2)
        attn    = (q @ k.transpose(-2,-1)) * self.scale
        mask_bf = mask.unsqueeze(1).expand(B,Freq,K,K).reshape(B*Freq,K,K)
        mask_bf = mask_bf.unsqueeze(1).expand(B*Freq,H,K,K)
        attn    = attn.masked_fill(~mask_bf, float("-inf"))
        attn    = torch.nan_to_num(torch.softmax(attn, dim=-1), nan=0.0)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        out = (attn @ v).transpose(1,2).reshape(B*Freq, K, C)
        out = self.out_proj(out).reshape(B, Freq, K, C).permute(0,2,3,1)
        return out

class SpectralFiLMLayer(nn.Module):
    def __init__(self, context_dim, bott_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(context_dim, context_dim), nn.PReLU(),
            nn.Linear(context_dim, bott_ch * 2))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, ctx, h):
        BK, C_bott, Freq, T = h.shape; B = ctx.shape[0]; K = BK // B
        ctx_flat = ctx.reshape(B*K, ctx.shape[2], Freq).permute(0,2,1)
        film = self.proj(ctx_flat)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = (1.0 + gamma).permute(0,2,1).unsqueeze(-1)
        beta  = beta.permute(0,2,1).unsqueeze(-1)
        return gamma * h + beta

class BottleneckFiLMBlock(nn.Module):
    def __init__(self, bott_ch, graph_dim, heads, dropout=0.0):
        super().__init__()
        self.pre_proj = nn.Linear(bott_ch, graph_dim)
        self.attn     = CrossNodeSpectralAttn(graph_dim, heads=heads, dropout=dropout)
        self.film     = SpectralFiLMLayer(graph_dim, bott_ch)

    def forward(self, h, A):
        BK, C, Freq, T = h.shape; B = A.shape[0]; K = BK // B
        spec = h.mean(dim=-1)
        proj = self.pre_proj(spec.permute(0,2,1)).permute(0,2,1)
        z    = proj.reshape(B, K, proj.shape[1], Freq)
        return self.film(self.attn(z, A), h)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: NeighborLogmagInjector  (for ablation iv)
# ─────────────────────────────────────────────────────────────────────────────

class NeighborLogmagInjector(nn.Module):
    """
    Pools neighbour log-magnitude to match a feature map's spatial resolution,
    then adds a learned 1×1 projection as a residual.

    nb_logmag : (BK, M, F_full, T_full)   full-resolution neighbour log-mag
    h         : (BK, C, F',    T'   )     encoder feature map to condition

    Zero-initialised projection → identity at init (no injection bias).
    """
    def __init__(self, n_mics: int, feat_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(n_mics, feat_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, nb_logmag: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        _, _, F_target, T_target = h.shape
        nb_pooled = F.adaptive_avg_pool2d(nb_logmag, (F_target, T_target))
        return h + self.proj(nb_pooled)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class GraphAblationV8(nn.Module):
    """
    Extends V8 (GraphInputCrossNodeV8) with two optional ablations:

    adj_mode:
      'fixed'    — use the stored per-scene adjacency (default, matches V8)
      'alltoall' — replace adj with fully-connected graph at runtime
                   (every node sees every other node's log-mag)

    inject_level1:
      False — neighbour log-mag injected only at input (matches V8)
      True  — also injected into encoder level-1 input via NeighborLogmagInjector
               (zero-init → starts identical to False)
    """
    def __init__(
        self,
        n_mics: int,
        n_fft: int          = 512,
        hop: int            = 128,
        base: int           = 32,
        depth: int          = 4,
        drop: float         = 0.0,
        graph_dim: int      = 128,
        graph_heads: int    = 4,
        graph_dropout: float= 0.0,
        use_graph: bool     = True,
        adj_mode: str       = "fixed",     # "fixed" | "alltoall"
        inject_level1: bool = False,
    ):
        super().__init__()
        assert depth >= 2
        assert adj_mode in ("fixed", "alltoall"), f"Unknown adj_mode: {adj_mode}"
        self.n_mics        = int(n_mics)
        self.use_graph     = bool(use_graph)
        self.adj_mode      = adj_mode
        self.inject_level1 = bool(inject_level1) and use_graph

        self.stft = TorchSTFT(n_fft=n_fft, hop=hop)

        # Input channels: 3M self-features + M neighbour log-mag (when graph ON)
        in_ch = (3 + 1) * n_mics if use_graph else 3 * n_mics

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc     = nn.ModuleList()
        self.enc_chs: List[int] = []
        c_in, c = in_ch, base
        for _ in range(depth):
            self.enc.append(DownBlock(c_in, c, drop=drop))
            self.enc_chs.append(c)
            c_in = c; c *= 2

        # ── Bottleneck ───────────────────────────────────────────────────────
        bott_ch      = self.enc_chs[-1] * 2
        self.bott_ch = bott_ch
        self.bot1    = ConvGNAct(self.enc_chs[-1], bott_ch, drop=drop)
        self.bot2    = ConvGNAct(bott_ch, bott_ch, drop=drop)

        if use_graph:
            self.bott_film = BottleneckFiLMBlock(
                bott_ch, graph_dim, graph_heads, graph_dropout)

        # ── Ablation (iv): level-1 injector ──────────────────────────────────
        # Applied to the downsampled output of encoder level 0
        # (shape: BK, enc_chs[0], F/2, T/2)
        if self.inject_level1:
            self.nb_injector_l1 = NeighborLogmagInjector(n_mics, self.enc_chs[0])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec = nn.ModuleList()
        c_in = bott_ch
        for i in reversed(range(depth)):
            self.dec.append(UpBlock(c_in, self.enc_chs[i], self.enc_chs[i], drop=drop))
            c_in = self.enc_chs[i]

        self.out_mask = nn.Conv2d(base, 1, 1)

    # ─── helpers ────────────────────────────────────────────────────────────

    def _self_features(self, X):
        mag    = X.abs().clamp_min(EPS)
        logmag = torch.log1p(mag)
        real   = X.real / (mag + EPS)
        imag   = X.imag / (mag + EPS)
        return torch.cat([logmag, real, imag], dim=1), X[:, 0]

    def _neighbor_logmag(self, X, A):
        """
        Adjacency-weighted average of neighbour log-magnitudes (self-loops excluded).
        X: (BK, M, F, T) complex;  A: (B, K, K)
        returns: (BK, M, F, T)
        """
        BK, M, Freq, TT = X.shape
        B = A.shape[0]; K = BK // B
        logmag = torch.log1p(X.abs().clamp_min(EPS)).reshape(B, K, M, Freq, TT)
        A_no_self = A * (1.0 - torch.eye(K, device=A.device, dtype=A.dtype).unsqueeze(0))
        row_sum   = A_no_self.sum(dim=-1, keepdim=True).clamp_min(EPS)
        A_norm    = A_no_self / row_sum                            # row-stochastic
        nb        = torch.einsum("bkj,bjmft->bkmft", A_norm, logmag)
        return nb.reshape(BK, M, Freq, TT)

    def _make_alltoall_adj(self, A: torch.Tensor) -> torch.Tensor:
        """Replace stored adj with fully-connected (ones minus diagonal)."""
        B, K, _ = A.shape
        eye = torch.eye(K, device=A.device, dtype=A.dtype).unsqueeze(0)
        return torch.ones(B, K, K, device=A.device, dtype=A.dtype) - eye

    # ─── forward ────────────────────────────────────────────────────────────

    def forward(self, Y: torch.Tensor, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K, M, T = Y.shape
        assert M == self.n_mics

        Yflat = Y.reshape(B * K, M, T)
        X     = self.stft.stft(Yflat)                          # (BK, M, F, TT)
        self_feat, X_ref = self._self_features(X)

        if self.use_graph:
            if A is None:
                raise ValueError("Adjacency A required when graph is enabled.")

            # ── Ablation (iii): optionally override adjacency ─────────────
            if self.adj_mode == "alltoall":
                A_used = self._make_alltoall_adj(A)
            else:
                A_used = A                                      # stored adj

            # ── Primary: input-level neighbour log-mag ────────────────────
            nb_logmag = self._neighbor_logmag(X, A_used)       # (BK, M, F, TT)
            feat      = torch.cat([self_feat, nb_logmag], dim=1)  # (BK, 4M, F, TT)
        else:
            A_used    = A
            nb_logmag = None
            feat      = self_feat

        # ── Encoder with optional level-1 injection ───────────────────────
        skips = []
        x     = feat
        for i, blk in enumerate(self.enc):
            s, x = blk(x)
            skips.append(s)

            # ── Ablation (iv): inject nb_logmag into level-1 input ────────
            # After encoder level 0 finishes, x is the strided output
            # (BK, enc_chs[0], F/2, T/2) — input to encoder level 1.
            if self.inject_level1 and i == 0:
                x = self.nb_injector_l1(nb_logmag, x)

        x = self.bot2(self.bot1(x))

        if self.use_graph:
            x = self.bott_film(x, A_used)

        for blk, skip in zip(self.dec, reversed(skips)):
            x = blk(x, skip)

        mask = torch.sigmoid(self.out_mask(x).squeeze(1))
        yhat = self.stft.istft(mask * X_ref, length=T).reshape(B, K, T)
        return yhat


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_one_epoch(model, loader, optimizer, device, epoch, is_train,
                  log_interval=20) -> Dict:
    model.train(is_train)
    total_loss = total_out = total_in = total_n = 0.0
    pbar = tqdm(loader,
                desc=(f"Train {epoch:03d}" if is_train else f"Val   {epoch:03d}"),
                ncols=110)
    for step, batch in enumerate(pbar, 1):
        Y = batch["Y"].to(device, non_blocking=True)
        S = batch["S"].to(device, non_blocking=True)
        A = batch["A"].to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            Yhat = model(Y, A)
            loss = si_sdr_loss(Yhat, S)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        with torch.no_grad():
            out = compute_si_sdr_metric(Yhat, S).mean().item()
            inp = compute_si_sdr_metric(Y[:, :, 0, :], S).mean().item()
        bsz         = Y.shape[0]
        total_loss += loss.item() * bsz
        total_out  += out * bsz; total_in += inp * bsz; total_n += bsz
        if step % log_interval == 0 or step == 1:
            pbar.set_postfix({
                "loss":  f"{total_loss/total_n:.3f}",
                "in":    f"{total_in/total_n:.2f}",
                "out":   f"{total_out/total_n:.2f}",
                "delta": f"{(total_out-total_in)/total_n:+.2f}",
            })
    n = max(total_n, 1)
    return {"loss": total_loss/n, "sisdr_in": total_in/n,
            "sisdr_out": total_out/n, "delta": (total_out-total_in)/n}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    # paths
    ap.add_argument("--data",        required=True)
    ap.add_argument("--out",         required=True)
    # optimisation
    ap.add_argument("--batch",       type=int,   default=8)
    ap.add_argument("--epochs",      type=int,   default=120)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--patience",    type=int,   default=20)
    ap.add_argument("--seed",        type=int,   default=1337)
    ap.add_argument("--num_workers", type=int,   default=6)
    ap.add_argument("--cache_shards_in_ram", action="store_true")
    # audio
    ap.add_argument("--crop_s",      type=float, default=2.0)
    ap.add_argument("--nfft",        type=int,   default=512)
    ap.add_argument("--hop",         type=int,   default=128)
    ap.add_argument("--n_mics_override", type=int, default=None)
    # model
    ap.add_argument("--base",        type=int,   default=32)
    ap.add_argument("--depth",       type=int,   default=4)
    ap.add_argument("--drop",        type=float, default=0.0)
    # graph
    ap.add_argument("--disable_graph", action="store_true",
                    help="strict local-only baseline (no cross-node path)")
    ap.add_argument("--graph_dim",   type=int,   default=128)
    ap.add_argument("--graph_heads", type=int,   default=4)
    ap.add_argument("--graph_dropout", type=float, default=0.0)
    # ablation flags
    ap.add_argument("--adj_mode",    default="fixed", choices=["fixed", "alltoall"],
                    help="(iii) fixed=stored adj, alltoall=fully-connected override")
    ap.add_argument("--inject_level1", action="store_true",
                    help="(iv) also inject neighbour log-mag at encoder level 1")
    ap.add_argument("--save_every",  type=int,   default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    crop_samples = int(args.crop_s * SR) if args.crop_s > 0 else None
    n_mics       = resolve_n_mics(args.data, args.n_mics_override)
    use_graph    = not args.disable_graph

    print(f"Device: {device}  |  graph: {'ON' if use_graph else 'OFF'}"
          f"  |  adj_mode: {args.adj_mode}"
          f"  |  inject_level1: {args.inject_level1}")

    train_ds = RestaurantSceneDataset(
        os.path.join(args.data, "train"), crop_samples, args.cache_shards_in_ram)
    val_ds   = RestaurantSceneDataset(
        os.path.join(args.data, "val"),   crop_samples, args.cache_shards_in_ram)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    model = GraphAblationV8(
        n_mics        = n_mics,
        n_fft         = args.nfft,
        hop           = args.hop,
        base          = args.base,
        depth         = args.depth,
        drop          = args.drop,
        graph_dim     = args.graph_dim,
        graph_heads   = args.graph_heads,
        graph_dropout = args.graph_dropout,
        use_graph     = use_graph,
        adj_mode      = args.adj_mode,
        inject_level1 = args.inject_level1,
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

    write_json(os.path.join(args.out, "args.json"), {
        **vars(args), "resolved_n_mics": n_mics, "graph_enabled": use_graph})

    history = []; best_val = float("inf"); best_epoch = 0; patience_cnt = 0

    for epoch in range(1, args.epochs + 1):
        train_m = run_one_epoch(model, train_loader, optimizer, device, epoch, True)
        val_m   = run_one_epoch(model, val_loader,   None,      device, epoch, False)
        scheduler.step(val_m["loss"])
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train Δ={train_m['delta']:+.3f}  val sisdr={val_m['sisdr_out']:.3f}"
              f" Δ={val_m['delta']:+.3f}  lr={lr_now:.2e}")

        history.append({
            "epoch": epoch, "lr": lr_now,
            "train_loss": train_m["loss"], "train_sisdr_in": train_m["sisdr_in"],
            "train_sisdr_out": train_m["sisdr_out"], "train_delta": train_m["delta"],
            "val_loss": val_m["loss"],   "val_sisdr_in": val_m["sisdr_in"],
            "val_sisdr_out": val_m["sisdr_out"],   "val_delta": val_m["delta"],
        })
        write_json(os.path.join(args.out, "history.json"), {"history": history})

        if val_m["loss"] < best_val:
            best_val = val_m["loss"]; best_epoch = epoch; patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            print(f"  ★ new best  val_loss={best_val:.4f}")
        else:
            patience_cnt += 1
            if args.patience > 0 and patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch}"); break

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.out, f"ckpt_ep{epoch:03d}.pt"))

    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
    write_json(os.path.join(args.out, "summary.json"), {
        "best_epoch": best_epoch, "best_val_loss": best_val,
        "graph_enabled": use_graph, "adj_mode": args.adj_mode,
        "inject_level1": args.inject_level1,
        "resolved_n_mics": n_mics, "num_params": count_parameters(model)})
    print(f"\nDone. Best epoch={best_epoch}, best_val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
