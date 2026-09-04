#!/usr/bin/env python3
"""
train_graph_bottleneck_unet_v6.py

RestaurantSim v6: clean graph ON/OFF comparison.

Core principle:
- Graph ON  : each table's encoded bottleneck participates in cross-node message passing.
              The graph-updated latent is fused into the bottleneck before the decoder.
- Graph OFF : STRICT local-only baseline.
              No pooling across nodes, no GNN, no FiLM, no node MLP replacement,
              no access to other tables at all.
              Forward path is simply:
                  table mic input -> encoder -> bottleneck -> decoder -> output

This script keeps ONE codebase and ONE model class, with a boolean flag deciding
whether cross-node graph fusion is active. That makes the ON/OFF comparison clean.

Expected dataset shard (.npz) keys:
  Y: (B,K,M,T) float32
  target_refclean: (B,K,T) float32
  adj: (B,K,K) float32

Notes:
- Supports variable microphones per table by reading manifest.json key "M"
  unless overridden by --n_mics_override.
- Uses SI-SDR loss on waveform output.
- Graph branch operates on node bottleneck summaries, then broadcasts back to a
  bottleneck feature map and adds it residual-style.
- When --disable_graph is used, the graph branch is completely skipped.

Example (graph ON):
  CUDA_VISIBLE_DEVICES=0 python train_graph_bottleneck_unet_v6.py \
    --data /home/rrame12/Desktop/Datasets/RestaurantSim_v5_graph_needed \
    --out  /home/rrame12/Desktop/Research/ASN/runs_graph_bottleneck_unet_v6_on \
    --batch 8 --epochs 60 --lr 3e-4 \
    --crop_s 2.0 --nfft 512 --hop 128 --num_workers 6 \
    --base 32 --depth 4 --drop 0.0 \
    --graph_dim 128 --graph_layers 2 --graph_heads 2 \
    --pool mean --alpha_init 1.0

Example (graph OFF / strict local-only):
  CUDA_VISIBLE_DEVICES=0 python train_graph_bottleneck_unet_v6.py \
    --data /home/rrame12/Desktop/Datasets/RestaurantSim_v5_graph_needed \
    --out  /home/rrame12/Desktop/Research/ASN/runs_graph_bottleneck_unet_v6_off \
    --batch 8 --epochs 60 --lr 3e-4 \
    --crop_s 2.0 --nfft 512 --hop 128 --num_workers 6 \
    --base 32 --depth 4 --drop 0.0 \
    --disable_graph
"""

import os
import glob
import json
import math
import time
import random
import argparse
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


SR = 16000
EPS = 1e-8


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable SI-SDR in dB.
    est, ref: (..., T)
    returns: (...,)
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)

    ref_energy = (ref * ref).sum(dim=-1, keepdim=True) + eps
    s_target = ((est * ref).sum(dim=-1, keepdim=True) / ref_energy) * ref
    e_noise = est - s_target

    ratio = (s_target * s_target).sum(dim=-1) / ((e_noise * e_noise).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def si_sdr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -compute_si_sdr(est, ref, eps=eps).mean()


@torch.no_grad()
def compute_si_sdr_metric(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return compute_si_sdr(est, ref, eps=eps)

def si_sdr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -compute_si_sdr(est, ref, eps=eps).mean()


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, obj: Dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Manifest helpers
# -----------------------------------------------------------------------------
def read_dataset_manifest(data_root: str) -> Dict:
    manifest_path = os.path.join(data_root, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found at: {manifest_path}")
    return read_json(manifest_path)


def resolve_n_mics(data_root: str, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)
    manifest = read_dataset_manifest(data_root)
    if "M" not in manifest:
        raise KeyError(f"'M' not found in {os.path.join(data_root, 'manifest.json')}")
    return int(manifest["M"])


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class RestaurantSceneDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir: str, crop_samples: Optional[int], cache_shards_in_ram: bool = False):
        super().__init__()
        self.crop_samples = crop_samples
        self.cache = cache_shards_in_ram

        self.shard_paths = sorted(glob.glob(os.path.join(split_dir, "shard_*.npz")))
        if not self.shard_paths:
            raise RuntimeError(f"No shards found in {split_dir}")

        self.index: List[Tuple[int, int]] = []
        self._shard_sizes: List[int] = []
        for sp in self.shard_paths:
            with np.load(sp) as d:
                B = d["Y"].shape[0]
            self._shard_sizes.append(B)

        for shard_id, B in enumerate(self._shard_sizes):
            for i in range(B):
                self.index.append((shard_id, i))

        self._ram_cache: Dict[int, Dict[str, np.ndarray]] = {}

    def __len__(self):
        return len(self.index)

    def _load_shard(self, shard_id: int) -> Dict[str, np.ndarray]:
        if self.cache and shard_id in self._ram_cache:
            return self._ram_cache[shard_id]

        sp = self.shard_paths[shard_id]
        d = np.load(sp)
        shard = {k: d[k] for k in d.files}
        d.close()

        if self.cache:
            self._ram_cache[shard_id] = shard
        return shard

    def __getitem__(self, idx: int):
        shard_id, local_i = self.index[idx]
        shard = self._load_shard(shard_id)

        Y = shard["Y"][local_i]               # (K,M,T)
        S = shard["target_refclean"][local_i] # (K,T)
        A = shard["adj"][local_i]             # (K,K)

        K, M, TT = Y.shape
        if self.crop_samples is not None and self.crop_samples < TT:
            start = np.random.randint(0, TT - self.crop_samples + 1)
            Y = Y[:, :, start:start + self.crop_samples]
            S = S[:, start:start + self.crop_samples]

        return {
            "Y": torch.from_numpy(Y).float(),
            "S": torch.from_numpy(S).float(),
            "A": torch.from_numpy(A).float(),
        }


def collate_fn(batch_list):
    return {
        "Y": torch.stack([b["Y"] for b in batch_list], dim=0),  # (B,K,M,T)
        "S": torch.stack([b["S"] for b in batch_list], dim=0),  # (B,K,T)
        "A": torch.stack([b["A"] for b in batch_list], dim=0),  # (B,K,K)
    }


# -----------------------------------------------------------------------------
# STFT helper
# -----------------------------------------------------------------------------
class TorchSTFT(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 128, win_length: Optional[int] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win_length = win_length if win_length is not None else n_fft
        self.register_buffer("window", torch.hann_window(self.win_length))

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,M,T)
        returns complex X: (N,M,F,TT)
        """
        N, M, T = x.shape
        x = x.reshape(N * M, T)
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        Freq, Frames = X.shape[-2], X.shape[-1]
        return X.reshape(N, M, Freq, Frames)

    def istft(self, X: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=length,
        )


# -----------------------------------------------------------------------------
# U-Net blocks
# -----------------------------------------------------------------------------
class ConvGNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1, drop: float = 0.0):
        super().__init__()
        ng = min(8, cout)
        while cout % ng != 0 and ng > 1:
            ng -= 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p),
            nn.GroupNorm(ng, cout),
            nn.PReLU(),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, cin: int, cout: int, drop: float = 0.0):
        super().__init__()
        self.c1 = ConvGNAct(cin, cout, drop=drop)
        self.c2 = ConvGNAct(cout, cout, drop=drop)
        self.down = nn.Conv2d(cout, cout, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.c2(self.c1(x))
        skip = x
        x = self.down(x)
        return skip, x


class UpBlock(nn.Module):
    def __init__(self, cin: int, skip_ch: int, cout: int, drop: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        self.c1 = ConvGNAct(cout + skip_ch, cout, drop=drop)
        self.c2 = ConvGNAct(cout, cout, drop=drop)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, max(dw, 0), 0, max(dh, 0)))
            x = x[..., :skip.shape[-2], :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        x = self.c2(self.c1(x))
        return x


# -----------------------------------------------------------------------------
# Pooling modules for node summaries
# -----------------------------------------------------------------------------
class MeanPool(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.out_dim = in_ch

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h.mean(dim=(-2, -1))


class MeanStdPool(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.out_dim = 2 * in_ch

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mu = h.mean(dim=(-2, -1))
        std = h.std(dim=(-2, -1), unbiased=False)
        return torch.cat([mu, std], dim=1)


class AttnPool(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.PReLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.out_dim = in_ch

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        BK, C, Freq, Frames = h.shape
        s = self.score(h).view(BK, -1)
        a = torch.softmax(s, dim=-1).view(BK, 1, Freq, Frames)
        return (a * h).sum(dim=(-2, -1))


def build_pool(pool_name: str, bott_ch: int) -> nn.Module:
    pool_name = pool_name.lower()
    if pool_name == "mean":
        return MeanPool(bott_ch)
    if pool_name == "meanstd":
        return MeanStdPool(bott_ch)
    if pool_name == "attn":
        return AttnPool(bott_ch)
    raise ValueError(f"Unknown pool type: {pool_name}")


# -----------------------------------------------------------------------------
# Graph blocks
# -----------------------------------------------------------------------------
class GATLayer(nn.Module):
    def __init__(self, din: int, dout: int, heads: int = 2, dropout: float = 0.0, leaky: float = 0.2):
        super().__init__()
        self.dout = dout
        self.heads = heads
        self.dropout = dropout
        self.W = nn.Linear(din, heads * dout, bias=False)
        self.a = nn.Parameter(torch.randn(heads, 2 * dout) * 0.02)
        self.bias = nn.Parameter(torch.zeros(heads * dout))
        self.leaky = leaky

    def forward(self, z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        z: (B,K,D)
        A: (B,K,K)
        returns: (B,K,heads*dout)
        """
        B, K, _ = z.shape
        A = A.clone()
        eye = torch.eye(K, device=A.device, dtype=A.dtype)[None, :, :]
        A = torch.clamp(A + eye, 0.0, 1.0)

        Wh = self.W(z).view(B, K, self.heads, self.dout)
        Wh_i = Wh[:, :, None, :, :].expand(B, K, K, self.heads, self.dout)
        Wh_j = Wh[:, None, :, :, :].expand(B, K, K, self.heads, self.dout)
        cat = torch.cat([Wh_i, Wh_j], dim=-1)

        e = (cat * self.a[None, None, None, :, :]).sum(dim=-1)
        e = F.leaky_relu(e, negative_slope=self.leaky)

        mask = (A[:, :, :, None] > 0.0)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(e, dim=2)
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = (alpha[..., None] * Wh_j).sum(dim=2)
        out = out.reshape(B, K, self.heads * self.dout) + self.bias
        return out


class GATStack(nn.Module):
    def __init__(self, din: int, dhid: int, dout: int, layers: int = 2, heads: int = 2, dropout: float = 0.0):
        super().__init__()
        mods = []
        in_d = din
        for li in range(layers):
            out_d = dout if li == layers - 1 else dhid
            mods.append(GATLayer(in_d, out_d, heads=heads, dropout=dropout))
            in_d = heads * out_d
        self.mods = nn.ModuleList(mods)

    def forward(self, z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = z
        for i, m in enumerate(self.mods):
            h = m(h, A)
            if i != len(self.mods) - 1:
                h = F.elu(h)
        return h


# -----------------------------------------------------------------------------
# Strict local-vs-graph bottleneck U-Net
# -----------------------------------------------------------------------------
class GraphBottleneckUNetV6(nn.Module):
    """
    Graph ON:
        input per node -> encoder -> bottleneck h_k
        pool(h_k) for all nodes -> GAT -> z'_k
        z'_k projected+broadcast to bottleneck map delta_h_k
        h'_k = h_k + alpha * delta_h_k
        decoder(h'_k) -> output_k

    Graph OFF:
        input per node -> encoder -> bottleneck h_k -> decoder(h_k) -> output_k

    Important:
    - In graph OFF there is absolutely no branch that reads other nodes.
    - This is the clean ablation the user asked for.
    """
    def __init__(
        self,
        n_mics: int,
        n_fft: int = 512,
        hop: int = 128,
        base: int = 32,
        depth: int = 4,
        drop: float = 0.0,
        graph_dim: int = 128,
        graph_layers: int = 2,
        graph_heads: int = 2,
        pool: str = "mean",
        use_graph: bool = True,
        alpha_init: float = 1.0,
        graph_dropout: float = 0.0,
    ):
        super().__init__()
        assert depth >= 2, "depth must be >= 2"
        self.n_mics = int(n_mics)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.depth = int(depth)
        self.use_graph = bool(use_graph)

        self.stft = TorchSTFT(n_fft=n_fft, hop=hop)

        # Input features: log-mag + phase proxy from all mics
        # per mic: [log|X|, real/norm, imag/norm] => 3*M channels
        in_ch = 3 * self.n_mics

        # Encoder
        enc = []
        chs = []
        c_in = in_ch
        c = base
        for _ in range(depth):
            enc.append(DownBlock(c_in, c, drop=drop))
            chs.append(c)
            c_in = c
            c *= 2
        self.enc = nn.ModuleList(enc)

        # Bottleneck
        bott_ch = chs[-1] * 2
        self.bot1 = ConvGNAct(chs[-1], bott_ch, drop=drop)
        self.bot2 = ConvGNAct(bott_ch, bott_ch, drop=drop)
        self.bott_ch = bott_ch

        # Graph branch (used only when use_graph=True at runtime)
        self.pool = build_pool(pool, bott_ch)
        pool_dim = self.pool.out_dim
        self.pre_graph = nn.Sequential(
            nn.Linear(pool_dim, graph_dim),
            nn.PReLU(),
        )
        self.gnn = GATStack(
            din=graph_dim,
            dhid=graph_dim,
            dout=graph_dim,
            layers=graph_layers,
            heads=graph_heads,
            dropout=graph_dropout,
        )
        self.post_graph = nn.Sequential(
            nn.Linear(graph_heads * graph_dim, graph_dim),
            nn.PReLU(),
            nn.Linear(graph_dim, bott_ch),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

        # Decoder
        dec = []
        c_in = bott_ch
        for i in reversed(range(depth)):
            dec.append(UpBlock(c_in, chs[i], chs[i], drop=drop))
            c_in = chs[i]
        self.dec = nn.ModuleList(dec)

        self.out_mask = nn.Conv2d(base, 1, kernel_size=1)

    def _make_features(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        X: (N,M,F,TT) complex
        returns:
            feat: (N, 3M, F, TT)
            X_ref: (N, F, TT) complex
        """
        mag = torch.abs(X).clamp_min(EPS)
        logmag = torch.log1p(mag)
        real = X.real / (mag + EPS)
        imag = X.imag / (mag + EPS)
        feat = torch.cat([logmag, real, imag], dim=1)
        X_ref = X[:, 0]  # reference mic
        return feat, X_ref

    def _encode(self, feat: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        skips = []
        x = feat
        for blk in self.enc:
            s, x = blk(x)
            skips.append(s)
        x = self.bot2(self.bot1(x))
        return skips, x

    def _apply_graph(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        h: (B,K,C,F,T)
        A: (B,K,K)
        returns updated bottleneck map with same shape.
        """
        B, K, C, Freq, Frames = h.shape
        hk = h.reshape(B * K, C, Freq, Frames)
        z = self.pool(hk).reshape(B, K, -1)
        z = self.pre_graph(z)
        z = self.gnn(z, A)
        z = self.post_graph(z)                  # (B,K,C)
        delta = z[:, :, :, None, None].expand(B, K, C, Freq, Frames)
        return h + self.alpha * delta

    def _decode(self, h: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = h
        for blk, skip in zip(self.dec, reversed(skips)):
            x = blk(x, skip)
        mask = torch.sigmoid(self.out_mask(x).squeeze(1))
        return mask

    def forward(self, Y: torch.Tensor, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Y: (B,K,M,T)
        A: (B,K,K)
        returns waveform estimate: (B,K,T)
        """
        B, K, M, T = Y.shape
        assert M == self.n_mics, f"Expected n_mics={self.n_mics}, got {M}"

        Yflat = Y.reshape(B * K, M, T)
        X = self.stft.stft(Yflat)  # (BK,M,F,TT)
        feat, X_ref = self._make_features(X)

        skips, h = self._encode(feat)  # h: (BK,C,F,TT)
        _, C, Freq, Frames = h.shape

        if self.use_graph:
            if A is None:
                raise ValueError("Adjacency A must be provided when graph is enabled.")
            h_nodes = h.reshape(B, K, C, Freq, Frames)
            h_nodes = self._apply_graph(h_nodes, A)
            h = h_nodes.reshape(B * K, C, Freq, Frames)
        # else: strict local-only path; do absolutely nothing with other nodes.

        mask = self._decode(h, skips)  # (BK,F,TT)
        Yhat_spec = mask * X_ref
        yhat = self.stft.istft(Yhat_spec, length=T).reshape(B, K, T)
        return yhat


# -----------------------------------------------------------------------------
# Training / validation loops
# -----------------------------------------------------------------------------
def run_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    epoch: int,
    is_train: bool,
    log_interval: int = 20,
) -> Dict[str, float]:
    model.train(is_train)

    total_loss = 0.0
    total_sisdr = 0.0
    total_in_sisdr = 0.0
    total_n = 0

    pbar = tqdm(loader, desc=(f"Train {epoch:03d}" if is_train else f"Val {epoch:03d}"), ncols=110)
    for step, batch in enumerate(pbar, start=1):
        Y = batch["Y"].to(device, non_blocking=True)  # (B,K,M,T)
        S = batch["S"].to(device, non_blocking=True)  # (B,K,T)
        A = batch["A"].to(device, non_blocking=True)  # (B,K,K)

        with torch.set_grad_enabled(is_train):
            Yhat = model(Y, A)
            loss = si_sdr_loss(Yhat, S)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        with torch.no_grad():
            sisdr_out = compute_si_sdr_metric(Yhat, S).mean().item()
            sisdr_in = compute_si_sdr_metric(Y[:, :, 0, :], S).mean().item()

        bsz = Y.shape[0]
        total_loss += float(loss.item()) * bsz
        total_sisdr += sisdr_out * bsz
        total_in_sisdr += sisdr_in * bsz
        total_n += bsz

        if step % log_interval == 0 or step == 1:
            pbar.set_postfix({
                "loss": f"{total_loss / max(total_n,1):.3f}",
                "in": f"{total_in_sisdr / max(total_n,1):.2f}",
                "out": f"{total_sisdr / max(total_n,1):.2f}",
                "delta": f"{(total_sisdr - total_in_sisdr) / max(total_n,1):+.2f}",
            })

    avg_loss = total_loss / max(total_n, 1)
    avg_out = total_sisdr / max(total_n, 1)
    avg_in = total_in_sisdr / max(total_n, 1)
    return {
        "loss": avg_loss,
        "sisdr_in": avg_in,
        "sisdr_out": avg_out,
        "delta": avg_out - avg_in,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    # paths
    ap.add_argument("--data", type=str, required=True, help="dataset root containing train/val/test and manifest.json")
    ap.add_argument("--out", type=str, required=True)

    # optimization
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--cache_shards_in_ram", action="store_true")

    # audio / crops
    ap.add_argument("--crop_s", type=float, default=2.0, help="crop length in seconds; <=0 means full length")
    ap.add_argument("--nfft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--n_mics_override", type=int, default=None)

    # model
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.0)

    # graph
    ap.add_argument("--disable_graph", action="store_true", help="strict local-only baseline")
    ap.add_argument("--graph_dim", type=int, default=128)
    ap.add_argument("--graph_layers", type=int, default=2)
    ap.add_argument("--graph_heads", type=int, default=2)
    ap.add_argument("--graph_dropout", type=float, default=0.0)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "meanstd", "attn"])
    ap.add_argument("--alpha_init", type=float, default=1.0)

    # schedule
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--save_every", type=int, default=0)

    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_samples = None if args.crop_s <= 0 else int(round(args.crop_s * SR))
    n_mics = resolve_n_mics(args.data, override=args.n_mics_override)

    print("=" * 80)
    print("RestaurantSim Graph Bottleneck U-Net v6")
    print(f"device           : {device}")
    print(f"data             : {args.data}")
    print(f"out              : {args.out}")
    print(f"resolved_n_mics  : {n_mics}")
    print(f"graph_enabled    : {not args.disable_graph}")
    print(f"crop_samples     : {crop_samples}")
    print("=" * 80)

    # datasets
    train_ds = RestaurantSceneDataset(
        split_dir=os.path.join(args.data, "train"),
        crop_samples=crop_samples,
        cache_shards_in_ram=args.cache_shards_in_ram,
    )
    val_ds = RestaurantSceneDataset(
        split_dir=os.path.join(args.data, "val"),
        crop_samples=crop_samples,
        cache_shards_in_ram=args.cache_shards_in_ram,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = GraphBottleneckUNetV6(
        n_mics=n_mics,
        n_fft=args.nfft,
        hop=args.hop,
        base=args.base,
        depth=args.depth,
        drop=args.drop,
        graph_dim=args.graph_dim,
        graph_layers=args.graph_layers,
        graph_heads=args.graph_heads,
        pool=args.pool,
        use_graph=(not args.disable_graph),
        alpha_init=args.alpha_init,
        graph_dropout=args.graph_dropout,
    ).to(device)

    print(f"trainable params : {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    args_to_save = vars(args).copy()
    args_to_save["resolved_n_mics"] = n_mics
    write_json(os.path.join(args.out, "args.json"), args_to_save)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    ckpt_last = os.path.join(args.out, "last.pt")
    ckpt_best = os.path.join(args.out, "best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_stats = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            is_train=True,
        )
        val_stats = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            epoch=epoch,
            is_train=False,
        )

        scheduler.step(val_stats["loss"])
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "lr": lr_now,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "time_sec": elapsed,
        }
        history.append(row)
        write_json(os.path.join(args.out, "history.json"), {"history": history})

        print(
            f"\nEpoch {epoch:03d} | "
            f"train loss {train_stats['loss']:.3f} | val loss {val_stats['loss']:.3f} | "
            f"train Δ {train_stats['delta']:+.2f} dB | val Δ {val_stats['delta']:+.2f} dB | "
            f"lr {lr_now:.2e} | time {elapsed/60.0:.1f} min"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": args_to_save,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "best_val": best_val,
            "history": history,
        }
        torch.save(state, ckpt_last)

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(state, os.path.join(args.out, f"epoch_{epoch:03d}.pt"))

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_epoch = epoch
            bad_epochs = 0
            state["best_val"] = best_val
            torch.save(state, ckpt_best)
            print(f"[best] saved to {ckpt_best}")
        else:
            bad_epochs += 1
            print(f"[no improvement] bad_epochs={bad_epochs}/{args.patience}")

        if bad_epochs >= args.patience:
            print("Early stopping triggered.")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "graph_enabled": not args.disable_graph,
        "resolved_n_mics": n_mics,
        "num_params": count_parameters(model),
    }
    write_json(os.path.join(args.out, "summary.json"), summary)

    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
