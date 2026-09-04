#!/usr/bin/env python3
"""
eval_graph_bottleneck_unet_v6_grouped.py

Evaluate Graph Bottleneck U-Net v6 on RestaurantSim with:
- overall SI-SDR_in / SI-SDR_out / Delta
- grouped metrics by local speaker count (1/2/3/...)
- grouped metrics by difficulty bins if metadata is available
- grouped metrics by local mic corruption if metadata is available
- optional saving of a few example wavs

This script is compatible with the v6 training checkpoint format and with
RestaurantSim datasets that save per-shard JSONL metadata alongside shard_XXXX.npz.

Expected metadata (if available in JSONL rows):
- speakers_per_table : list[int] of length K
- hard_flags or hard_tables or per-table hardness-like fields (optional)
- local_mic_corruption / mic_corruption / corruption-like fields (optional)
- local_frac / local_energy_fraction / bleed_sir_db / per-table variants (optional)

The script is defensive: if a given metadata field is absent, that grouping is skipped.
"""

import os
import glob
import json
import argparse
from collections import defaultdict
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm

SR = 16000
EPS = 1e-8


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, obj: Dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def to_python(x: Any):
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def compute_si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    ref_energy = (ref * ref).sum(dim=-1, keepdim=True) + eps
    s_target = ((est * ref).sum(dim=-1, keepdim=True) / ref_energy) * ref
    e_noise = est - s_target
    ratio = (s_target * s_target).sum(dim=-1) / ((e_noise * e_noise).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


# -----------------------------------------------------------------------------
# Manifest helpers
# -----------------------------------------------------------------------------
def read_dataset_manifest(data_root: str) -> Dict:
    manifest_path = os.path.join(data_root, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found at: {manifest_path}")
    return read_json(manifest_path)


def resolve_n_mics(data_root: str, ckpt_args: Dict, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)
    if "resolved_n_mics" in ckpt_args:
        return int(ckpt_args["resolved_n_mics"])
    if "n_mics_override" in ckpt_args and ckpt_args["n_mics_override"] is not None:
        return int(ckpt_args["n_mics_override"])
    manifest = read_dataset_manifest(data_root)
    if "M" not in manifest:
        raise KeyError(f"'M' not found in {os.path.join(data_root, 'manifest.json')}")
    return int(manifest["M"])


# -----------------------------------------------------------------------------
# Dataset with metadata
# -----------------------------------------------------------------------------
class RestaurantSceneDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir: str, crop_samples: Optional[int], cache: bool = True):
        super().__init__()
        self.crop_samples = crop_samples
        self.cache = cache

        self.shard_paths = sorted(glob.glob(os.path.join(split_dir, "shard_*.npz")))
        if not self.shard_paths:
            raise RuntimeError(f"No shards found in {split_dir}")

        self.meta_paths = [sp.replace(".npz", ".jsonl") for sp in self.shard_paths]

        self.index: List[Tuple[int, int]] = []
        self._sizes: List[int] = []
        for sp in self.shard_paths:
            with np.load(sp) as d:
                B = d["Y"].shape[0]
            self._sizes.append(B)
        for sid, B in enumerate(self._sizes):
            for i in range(B):
                self.index.append((sid, i))

        self._cache_npz: Dict[int, Dict[str, np.ndarray]] = {}
        self._cache_meta: Dict[int, list] = {}

    def __len__(self):
        return len(self.index)

    def _load_npz(self, sid: int) -> Dict[str, np.ndarray]:
        if self.cache and sid in self._cache_npz:
            return self._cache_npz[sid]
        d = np.load(self.shard_paths[sid])
        sh = {k: d[k] for k in d.files}
        d.close()
        if self.cache:
            self._cache_npz[sid] = sh
        return sh

    def _load_meta(self, sid: int):
        mp = self.meta_paths[sid]
        if not os.path.exists(mp):
            return None
        if self.cache and sid in self._cache_meta:
            return self._cache_meta[sid]
        rows = []
        with open(mp, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if self.cache:
            self._cache_meta[sid] = rows
        return rows

    def __getitem__(self, idx: int):
        sid, li = self.index[idx]
        sh = self._load_npz(sid)
        meta_rows = self._load_meta(sid)

        Y = sh["Y"][li]               # (K,M,T)
        S = sh["target_refclean"][li] # (K,T)
        A = sh["adj"][li].astype(np.float32)

        K, M, TT = Y.shape
        if self.crop_samples is not None and self.crop_samples < TT:
            start = np.random.randint(0, TT - self.crop_samples + 1)
            Y = Y[:, :, start:start + self.crop_samples]
            S = S[:, start:start + self.crop_samples]

        row = None
        if meta_rows is not None and li < len(meta_rows):
            row = meta_rows[li]

        return {
            "Y": torch.from_numpy(Y).float(),
            "S": torch.from_numpy(S).float(),
            "A": torch.from_numpy(A).float(),
            "meta": row,
            "scene_id": f"shard{sid:04d}_item{li:04d}",
        }


def collate_fn(batch):
    return {
        "Y": torch.stack([b["Y"] for b in batch], 0),
        "S": torch.stack([b["S"] for b in batch], 0),
        "A": torch.stack([b["A"] for b in batch], 0),
        "meta": [b["meta"] for b in batch],
        "scene_id": [b["scene_id"] for b in batch],
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
# Model blocks (must match training v6)
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
        return self.c2(self.c1(x))


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


class GraphBottleneckUNetV6(nn.Module):
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
        self.n_mics = int(n_mics)
        self.use_graph = bool(use_graph)
        self.stft = TorchSTFT(n_fft=n_fft, hop=hop)

        in_ch = 3 * self.n_mics
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

        bott_ch = chs[-1] * 2
        self.bot1 = ConvGNAct(chs[-1], bott_ch, drop=drop)
        self.bot2 = ConvGNAct(bott_ch, bott_ch, drop=drop)
        self.bott_ch = bott_ch

        self.pool = build_pool(pool, bott_ch)
        pool_dim = self.pool.out_dim
        self.pre_graph = nn.Sequential(nn.Linear(pool_dim, graph_dim), nn.PReLU())
        self.gnn = GATStack(din=graph_dim, dhid=graph_dim, dout=graph_dim, layers=graph_layers, heads=graph_heads, dropout=graph_dropout)
        self.post_graph = nn.Sequential(
            nn.Linear(graph_heads * graph_dim, graph_dim),
            nn.PReLU(),
            nn.Linear(graph_dim, bott_ch),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

        dec = []
        c_in = bott_ch
        for i in reversed(range(depth)):
            dec.append(UpBlock(c_in, chs[i], chs[i], drop=drop))
            c_in = chs[i]
        self.dec = nn.ModuleList(dec)
        self.out_mask = nn.Conv2d(base, 1, kernel_size=1)

    def _make_features(self, X: torch.Tensor):
        mag = torch.abs(X).clamp_min(EPS)
        logmag = torch.log1p(mag)
        real = X.real / (mag + EPS)
        imag = X.imag / (mag + EPS)
        feat = torch.cat([logmag, real, imag], dim=1)
        X_ref = X[:, 0]
        return feat, X_ref

    def _encode(self, feat: torch.Tensor):
        skips = []
        x = feat
        for blk in self.enc:
            s, x = blk(x)
            skips.append(s)
        x = self.bot2(self.bot1(x))
        return skips, x

    def _apply_graph(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, K, C, Freq, Frames = h.shape
        hk = h.reshape(B * K, C, Freq, Frames)
        z = self.pool(hk).reshape(B, K, -1)
        z = self.pre_graph(z)
        z = self.gnn(z, A)
        z = self.post_graph(z)
        delta = z[:, :, :, None, None].expand(B, K, C, Freq, Frames)
        return h + self.alpha * delta

    def _decode(self, h: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = h
        for blk, skip in zip(self.dec, reversed(skips)):
            x = blk(x, skip)
        mask = torch.sigmoid(self.out_mask(x).squeeze(1))
        return mask

    def forward(self, Y: torch.Tensor, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K, M, T = Y.shape
        assert M == self.n_mics
        Yflat = Y.reshape(B * K, M, T)
        X = self.stft.stft(Yflat)
        feat, X_ref = self._make_features(X)
        skips, h = self._encode(feat)
        _, C, Freq, Frames = h.shape

        if self.use_graph:
            if A is None:
                raise ValueError("Adjacency A must be provided when graph is enabled.")
            h_nodes = h.reshape(B, K, C, Freq, Frames)
            h_nodes = self._apply_graph(h_nodes, A)
            h = h_nodes.reshape(B * K, C, Freq, Frames)

        mask = self._decode(h, skips)
        Yhat_spec = mask * X_ref
        yhat = self.stft.istft(Yhat_spec, length=T).reshape(B, K, T)
        return yhat


# -----------------------------------------------------------------------------
# Metadata parsing / grouping helpers
# -----------------------------------------------------------------------------
def extract_list_per_table(meta: Optional[Dict], keys: List[str], K: int):
    if meta is None:
        return None
    for k in keys:
        if k in meta and isinstance(meta[k], list) and len(meta[k]) == K:
            return meta[k]
    return None


def extract_scalar_or_list_per_table(meta: Optional[Dict], per_table_keys: List[str], scalar_keys: List[str], K: int):
    vals = extract_list_per_table(meta, per_table_keys, K)
    if vals is not None:
        return vals
    if meta is not None:
        for k in scalar_keys:
            if k in meta:
                return [meta[k]] * K
    return None


def infer_corruption_per_table(meta: Optional[Dict], K: int):
    if meta is None:
        return None

    vals = extract_list_per_table(meta,
                                  ["local_mic_corruption", "mic_corruption", "corrupted_tables", "table_corruption_flags"],
                                  K)
    if vals is not None:
        out = []
        for v in vals:
            if isinstance(v, bool):
                out.append(v)
            elif isinstance(v, (int, float)):
                out.append(bool(v))
            elif isinstance(v, dict):
                out.append(True)
            else:
                out.append(v is not None)
        return out

    cm = meta.get("corruption_meta", None)
    if isinstance(cm, dict) and "events" in cm and isinstance(cm["events"], list):
        flags = [False] * K
        for ev in cm["events"]:
            if isinstance(ev, dict) and "table" in ev:
                tk = int(ev["table"])
                if 0 <= tk < K:
                    flags[tk] = True
        return flags

    if "corruption_applied" in meta:
        return [bool(meta["corruption_applied"])] * K
    if "local_mic_corruption_applied" in meta:
        return [bool(meta["local_mic_corruption_applied"])] * K
    return None


def infer_hard_flags(meta: Optional[Dict], K: int):
    if meta is None:
        return None

    vals = extract_list_per_table(meta,
                                  ["hard_flags", "hard_tables", "table_hard_flags", "is_hard_table"],
                                  K)
    if vals is not None:
        return [bool(v) for v in vals]

    # derive from local fraction or bleed SIR if present
    local_frac = extract_list_per_table(meta,
                                        ["local_frac", "local_fractions", "local_energy_fraction", "local_energy_fractions"],
                                        K)
    if local_frac is not None:
        return [float(v) < 0.75 for v in local_frac]

    bleed_sir = extract_list_per_table(meta,
                                       ["bleed_sir_db", "bleed_sir_dbs", "cross_bleed_sir_db", "cross_bleed_sir_dbs"],
                                       K)
    if bleed_sir is not None:
        return [float(v) < 6.0 for v in bleed_sir]

    return None


def infer_difficulty_bucket_per_table(meta: Optional[Dict], K: int):
    if meta is None:
        return None

    local_frac = extract_list_per_table(meta,
                                        ["local_frac", "local_fractions", "local_energy_fraction", "local_energy_fractions"],
                                        K)
    if local_frac is not None:
        buckets = []
        for v in local_frac:
            x = float(v)
            if x < 0.50:
                buckets.append("hard")
            elif x < 0.75:
                buckets.append("medium")
            else:
                buckets.append("easy")
        return buckets

    bleed_sir = extract_list_per_table(meta,
                                       ["bleed_sir_db", "bleed_sir_dbs", "cross_bleed_sir_db", "cross_bleed_sir_dbs"],
                                       K)
    if bleed_sir is not None:
        buckets = []
        for v in bleed_sir:
            x = float(v)
            if x < 0.0:
                buckets.append("hard")
            elif x < 6.0:
                buckets.append("medium")
            else:
                buckets.append("easy")
        return buckets

    hard_flags = infer_hard_flags(meta, K)
    if hard_flags is not None:
        return ["hard" if f else "nonhard" for f in hard_flags]

    return None


def make_group_summary(rows: List[Dict]) -> Dict[str, Any]:
    if len(rows) == 0:
        return {"n": 0}
    arr_in = np.asarray([r["sisdr_in"] for r in rows], dtype=np.float64)
    arr_out = np.asarray([r["sisdr_out"] for r in rows], dtype=np.float64)
    arr_delta = np.asarray([r["delta"] for r in rows], dtype=np.float64)
    return {
        "n": int(len(rows)),
        "sisdr_in_mean": float(arr_in.mean()),
        "sisdr_in_std": float(arr_in.std()),
        "sisdr_out_mean": float(arr_out.mean()),
        "sisdr_out_std": float(arr_out.std()),
        "delta_mean": float(arr_delta.mean()),
        "delta_std": float(arr_delta.std()),
    }


# -----------------------------------------------------------------------------
# Main eval
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--full_length", type=int, default=1)
    ap.add_argument("--crop_s", type=float, default=0.0)
    ap.add_argument("--save_scenes", type=int, default=0)
    ap.add_argument("--n_mics_override", type=int, default=None)
    args = ap.parse_args()

    ensure_dir(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    crop_samples = None if int(args.full_length) == 1 else int(round(args.crop_s * SR))
    n_mics = resolve_n_mics(args.data, ckpt_args, override=args.n_mics_override)
    graph_enabled = not bool(ckpt_args.get("disable_graph", False))

    print("=" * 80)
    print("Evaluate Graph Bottleneck U-Net v6 (grouped)")
    print(f"device          : {device}")
    print(f"data            : {args.data}")
    print(f"split           : {args.split}")
    print(f"ckpt            : {args.ckpt}")
    print(f"out             : {args.out}")
    print(f"graph_enabled   : {graph_enabled}")
    print(f"resolved_n_mics : {n_mics}")
    print(f"crop_samples    : {crop_samples}")
    print("=" * 80)

    ds = RestaurantSceneDataset(
        split_dir=os.path.join(args.data, args.split),
        crop_samples=crop_samples,
        cache=True,
    )
    loader = torch.utils.data.DataLoader(
        ds,
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
        n_fft=int(ckpt_args.get("nfft", 512)),
        hop=int(ckpt_args.get("hop", 128)),
        base=int(ckpt_args.get("base", 32)),
        depth=int(ckpt_args.get("depth", 4)),
        drop=float(ckpt_args.get("drop", 0.0)),
        graph_dim=int(ckpt_args.get("graph_dim", 128)),
        graph_layers=int(ckpt_args.get("graph_layers", 2)),
        graph_heads=int(ckpt_args.get("graph_heads", 2)),
        pool=str(ckpt_args.get("pool", "mean")),
        use_graph=graph_enabled,
        alpha_init=float(ckpt_args.get("alpha_init", 1.0)),
        graph_dropout=float(ckpt_args.get("graph_dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    overall_rows: List[Dict] = []
    by_spk = defaultdict(list)
    by_difficulty = defaultdict(list)
    by_corruption = defaultdict(list)
    saved = 0

    pbar = tqdm(loader, desc="Evaluating", ncols=110)
    with torch.no_grad():
        for batch in pbar:
            Y = batch["Y"].to(device, non_blocking=True)
            S = batch["S"].to(device, non_blocking=True)
            A = batch["A"].to(device, non_blocking=True)
            metas = batch["meta"]
            scene_ids = batch["scene_id"]

            Yhat = model(Y, A)
            sisdr_out = compute_si_sdr(Yhat, S)          # (B,K)
            sisdr_in = compute_si_sdr(Y[:, :, 0, :], S)  # (B,K)
            delta = sisdr_out - sisdr_in

            B, K = sisdr_out.shape

            pbar.set_postfix({"delta": f"{delta.mean().item():+.2f} dB"})

            for b in range(B):
                meta = metas[b]
                sid = scene_ids[b]

                spk_counts = extract_list_per_table(meta, ["speakers_per_table"], K)
                diff_buckets = infer_difficulty_bucket_per_table(meta, K)
                hard_flags = infer_hard_flags(meta, K)
                corrupt_flags = infer_corruption_per_table(meta, K)

                for k in range(K):
                    row = {
                        "scene_id": sid,
                        "table": int(k),
                        "sisdr_in": float(sisdr_in[b, k].item()),
                        "sisdr_out": float(sisdr_out[b, k].item()),
                        "delta": float(delta[b, k].item()),
                    }

                    if spk_counts is not None:
                        row["n_spk_local"] = int(spk_counts[k])
                    if diff_buckets is not None:
                        row["difficulty_bucket"] = str(diff_buckets[k])
                    if hard_flags is not None:
                        row["is_hard"] = bool(hard_flags[k])
                    if corrupt_flags is not None:
                        row["local_mic_corrupted"] = bool(corrupt_flags[k])

                    # optional carry-through of useful raw metadata if present
                    for maybe_key in [
                        "local_frac", "local_fractions", "local_energy_fraction", "local_energy_fractions",
                        "bleed_sir_db", "bleed_sir_dbs", "cross_bleed_sir_db", "cross_bleed_sir_dbs",
                    ]:
                        if meta is not None and maybe_key in meta:
                            v = meta[maybe_key]
                            if isinstance(v, list) and len(v) == K:
                                row[maybe_key.rstrip('s')] = to_python(v[k])
                            else:
                                row[maybe_key] = to_python(v)

                    overall_rows.append(row)

                    if "n_spk_local" in row:
                        by_spk[str(row["n_spk_local"])].append(row)
                    if "difficulty_bucket" in row:
                        by_difficulty[str(row["difficulty_bucket"])].append(row)
                    if "local_mic_corrupted" in row:
                        by_corruption["corrupted" if row["local_mic_corrupted"] else "clean"].append(row)

                if saved < args.save_scenes:
                    exdir = os.path.join(args.out, "examples")
                    ensure_dir(exdir)
                    for k in range(K):
                        sf.write(os.path.join(exdir, f"{sid}_table{k}_mix_ref.wav"), Y[b, k, 0].detach().cpu().numpy(), SR)
                        sf.write(os.path.join(exdir, f"{sid}_table{k}_target.wav"), S[b, k].detach().cpu().numpy(), SR)
                        sf.write(os.path.join(exdir, f"{sid}_table{k}_enh.wav"), Yhat[b, k].detach().cpu().numpy(), SR)
                    saved += 1

    overall_summary = make_group_summary(overall_rows)
    by_spk_summary = {k: make_group_summary(v) for k, v in sorted(by_spk.items(), key=lambda kv: kv[0])}
    by_difficulty_summary = {k: make_group_summary(v) for k, v in sorted(by_difficulty.items(), key=lambda kv: kv[0])}
    by_corruption_summary = {k: make_group_summary(v) for k, v in sorted(by_corruption.items(), key=lambda kv: kv[0])}

    summary = {
        "mode": "GRAPH_ON" if graph_enabled else "GRAPH_OFF",
        "ckpt": args.ckpt,
        "split": args.split,
        "overall": overall_summary,
        "by_n_spk_local": by_spk_summary,
        "by_difficulty": by_difficulty_summary,
        "by_local_mic_corruption": by_corruption_summary,
        "num_rows": len(overall_rows),
    }

    write_json(os.path.join(args.out, "metrics_summary.json"), summary)
    write_json(os.path.join(args.out, "metrics_per_table.json"), {"rows": overall_rows})

    print("\n=== TEST RESULTS ===")
    print(f"mode: {summary['mode']}")
    print(f"ckpt: {args.ckpt}")
    print(f"SI-SDR_in : {overall_summary.get('sisdr_in_mean', float('nan')):.2f} dB")
    print(f"SI-SDR_out: {overall_summary.get('sisdr_out_mean', float('nan')):.2f} dB")
    print(f"Delta     : {overall_summary.get('delta_mean', float('nan')):+.2f} dB")

    if len(by_spk_summary) > 0:
        print("\nBy local speaker count:")
        for k, v in by_spk_summary.items():
            print(f"  {k} spk | n={v['n']:4d} | in={v['sisdr_in_mean']:.2f} | out={v['sisdr_out_mean']:.2f} | Δ={v['delta_mean']:+.2f}")

    if len(by_difficulty_summary) > 0:
        print("\nBy difficulty:")
        for k, v in by_difficulty_summary.items():
            print(f"  {k:8s} | n={v['n']:4d} | in={v['sisdr_in_mean']:.2f} | out={v['sisdr_out_mean']:.2f} | Δ={v['delta_mean']:+.2f}")

    if len(by_corruption_summary) > 0:
        print("\nBy local mic corruption:")
        for k, v in by_corruption_summary.items():
            print(f"  {k:10s} | n={v['n']:4d} | in={v['sisdr_in_mean']:.2f} | out={v['sisdr_out_mean']:.2f} | Δ={v['delta_mean']:+.2f}")


if __name__ == "__main__":
    main()
